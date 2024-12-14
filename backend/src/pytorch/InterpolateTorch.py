import torch
import torch.nn.functional as F
import numpy as np
from abc import ABCMeta, abstractmethod
from queue import Queue
#from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .InterpolateArchs.DetectInterpolateArch import ArchDetect
from .UpscaleTorch import UpscalePytorch
import math
import os
import logging
import gc
import sys
from ..utils.Util import (
    printAndLog,
    errorAndLog,
    check_bfloat16_support,
    warnAndLog,
    log
)
from ..constants import HAS_SYSTEM_CUDA
from time import sleep

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)


class BaseInterpolate(metaclass=ABCMeta):
    @abstractmethod
    def _load(self):
        """Loads in the model"""
        self.stream = torch.cuda.Stream()
        self.prepareStream = torch.cuda.Stream()
        self.frame0 = None
        self.encode0 = None
        self.flownet = None
        self.encode = None
        self.tenFlow_div = None
        self.backwarp_tenGrid = None
        self.doEncodingOnFrame = False # set this by default
        self.CompareNet = None

    def handlePrecision(self, precision) -> torch.dtype:
        if precision == "auto":
            return torch.float16 if check_bfloat16_support() else torch.float32
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16
        if precision == "bfloat16":
            return torch.bfloat16

    @torch.inference_mode()
    def copyTensor(self, tensorToCopy: torch.Tensor, tensorCopiedTo: torch.Tensor):
        with torch.cuda.stream(self.stream):
            tensorToCopy.copy_(tensorCopiedTo, non_blocking=True)
        self.stream.synchronize()

    def hotUnload(self):
        self.flownet = None
        self.encode = None
        self.tenFlow_div = None
        self.backwarp_tenGrid = None
        self.f0encode = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

    @torch.inference_mode()
    def hotReload(self):
        self._load()

    @torch.inference_mode()
    def dynamicScaleCalculation(self,frame1):
        closest_value = None
        if self.CompareNet is not None: # when there is dynamic optical flow scaling enabled.
            ssim:torch.Tensor = self.CompareNet(self.frame0, frame1)
            possible_values = {0.25:1.5, 0.5:1.25, 1.0:1.0} # closest_value:representative_scale
            closest_value = min(possible_values, key=lambda v: abs(ssim.item() - v))
            closest_value = possible_values[closest_value]
        return closest_value

    @abstractmethod
    @torch.inference_mode()
    def __call__(self, img1, writeQueue:Queue, transition=False, upscaleModel:torch.nn.Module = None):
        """Perform processing"""

    @torch.inference_mode()
    def norm(self, frame: torch.Tensor):
        return (
            frame.reshape(self.height, self.width, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255.0)
        )

    @torch.inference_mode()
    def frame_to_tensor(self, frame) -> torch.Tensor:
        with torch.cuda.stream(self.prepareStream):
            frame = self.norm(
                torch.frombuffer(
                    frame,
                    dtype=torch.uint8,
                ).to(device=self.device, dtype=self.dtype, non_blocking=True)
            )
            frame = F.pad(frame, self.padding)

        self.prepareStream.synchronize()
        return frame

    @torch.inference_mode()
    def uncacheFrame(self):
        self.f0encode = None
        self.img0 = None

    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor):
        return frame.squeeze_(0).permute(1, 2, 0).mul(255).float().byte().contiguous().cpu().numpy()


class InterpolateGIMMTorch(BaseInterpolate):
    @torch.inference_mode()
    def __init__(
        self,
        modelPath: str,
        ceilInterpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        device: str = "default",
        dtype: str = "auto",
        backend: str = "pytorch",
        UHDMode: bool = False,
        *args,
        **kwargs,
    ):
        if device == "default":
            if torch.cuda.is_available():
                device = torch.device(
                    "cuda", 0
                )  # 0 is the device index, may have to change later
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)

        printAndLog("Using device: " + str(device))

        self.interpolateModel = modelPath
        self.width = width
        self.height = height
        self.device = device
        self.dtype = self.handlePrecision(dtype)
        self.backend = backend
        self.ceilInterpolateFactor = ceilInterpolateFactor
        self.frame0 = None
        self.scale = 0.5 # GIMM uses fat amounts of vram, needs really low flow resolution for regular resolutions
        if UHDMode:
            self.scale = 0.25 # GIMM uses fat amounts of vram, needs really low flow resolution for UHD
        self.doEncodingOnFrame = False
        
        self._load()

    @torch.inference_mode()
    def _load(self):
        self.stream = torch.cuda.Stream()
        self.prepareStream = torch.cuda.Stream()
        with torch.cuda.stream(self.prepareStream):
            from .InterpolateArchs.GIMM.gimmvfi_r import GIMMVFI_R

            self.flownet = GIMMVFI_R(
                model_path=self.interpolateModel, width=self.width, height=self.height
            )
            state_dict = torch.load(self.interpolateModel, map_location=self.device)["gimmvfi_r"]
            self.flownet.load_state_dict(state_dict)
            self.flownet.eval().to(device=self.device, dtype=self.dtype)

            _pad = 64
            tmp = max(_pad, int(_pad / self.scale))
            self.pw = math.ceil(self.width / tmp) * tmp
            self.ph = math.ceil(self.height / tmp) * tmp
            self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
            
            dummyInput = torch.zeros([1, 3, self.ph, self.pw], dtype=self.dtype, device=self.device)
            dummyInput2 = torch.zeros([1, 3, self.ph, self.pw], dtype=self.dtype, device=self.device)
            xs = torch.cat((dummyInput.unsqueeze(2), dummyInput2.unsqueeze(2)), dim=2).to(self.device, non_blocking=True)
            s_shape = xs.shape[-2:]
            
            # caching the timestep tensor in a dict with the timestep as a float for the key
            
            self.timestepDict = {}
            self.coordDict = {}

            for n in range(self.ceilInterpolateFactor):
                timestep = n / (self.ceilInterpolateFactor)
                timestep_tens = n * 1 /  self.ceilInterpolateFactor * torch.ones(xs.shape[0]).to(xs.device).to(self.dtype).reshape(-1, 1, 1, 1)
                self.timestepDict[timestep] = timestep_tens
                coord = (self.flownet.sample_coord_input(
                        1,
                        s_shape,
                        [1 / self.ceilInterpolateFactor * n],
                        device=self.device,
                        upsample_ratio=self.scale,
                ).to(non_blocking=True, dtype=self.dtype, device=self.device),None)
                self.coordDict[timestep] = coord
            
            log("GIMM loaded")
            log("Scale: " + str(self.scale))
            log("Using System CUDA: " + str(HAS_SYSTEM_CUDA))
            if not HAS_SYSTEM_CUDA:
                print("WARNING: System CUDA not found, falling back to PyTorch softsplat. This will be a bit slower.",file=sys.stderr)
            if self.backend == "tensorrt":
                warnAndLog(
                    "TensorRT is not implemented for GIMM yet, falling back to PyTorch"
                )
        self.prepareStream.synchronize()
    
    @torch.inference_mode()
    def __call__(self, img1, writeQueue:Queue, transition=False, upscaleModel:UpscalePytorch = None):
        if self.frame0 is None:
            self.frame0 = self.frame_to_tensor(img1)
            self.stream.synchronize()
            return
        frame1 = self.frame_to_tensor(img1)
        with torch.cuda.stream(self.stream):
            for n in range(self.ceilInterpolateFactor-1):
                if not transition:
                    timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                    coord = self.coordDict[timestep]
                    timestep_tens = self.timestepDict[timestep]
                    xs = torch.cat((self.frame0.unsqueeze(2), frame1.unsqueeze(2)), dim=2).to(
                    self.device, non_blocking=True,dtype=self.dtype
                    )

                    while self.flownet is None:
                        sleep(1)
                    
                    output = self.flownet(xs, coord, timestep_tens, ds_factor=self.scale)
                    
                    if torch.isnan(output).any():
                        # if there are nans in output, reload with float32 precision and process.... dumb fix but whatever
                        print("NaNs in output, returning the first image",file=sys.stderr)
                        if upscaleModel is not None:
                            img1 = upscaleModel(upscaleModel.frame_to_tensor(self.tensor_to_frame(img1)))
                        writeQueue.put(img1)

                    else:
                        if upscaleModel is not None:
                            output = upscaleModel(upscaleModel.frame_to_tensor(self.tensor_to_frame(output)))
                        else:
                            output = self.tensor_to_frame(output)
                        writeQueue.put(output)
                
                else:
                    if upscaleModel is not None:
                            img1 = upscaleModel(frame1[:, :, : self.height, : self.width])
                    writeQueue.put(img1)    
            self.copyTensor(self.frame0, frame1)

        self.stream.synchronize()

class InterpolateGMFSSTorch(BaseInterpolate):
    @torch.inference_mode()
    def __init__(
        self,
        modelPath: str,
        ceilInterpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        device: str = "default",
        dtype: str = "auto",
        backend: str = "pytorch",
        UHDMode: bool = False,
        ensemble: bool = False,
        dynamicScaledOpticalFlow: bool = False,
        *args,
        **kwargs,
    ):
        if device == "default":
            if torch.cuda.is_available():
                device = torch.device(
                    "cuda", 0
                )  # 0 is the device index, may have to change later
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)

        printAndLog("Using device: " + str(device))

        self.frame0 = None
        self.interpolateModel = modelPath
        self.width = width
        self.height = height
        self.device = device
        self.dtype = self.handlePrecision(dtype)
        self.backend = backend
        self.ceilInterpolateFactor = ceilInterpolateFactor
        # set up streams for async processing
        self.scale = 1
        self.ensemble = ensemble
        self.dynamicScaledOpticalFlow = dynamicScaledOpticalFlow
        self.UHDMode = UHDMode
        if UHDMode:
            self.scale = 0.5
        self._load()

    @torch.inference_mode()
    def _load(self):
        self.stream = torch.cuda.Stream()
        self.prepareStream = torch.cuda.Stream()
        with torch.cuda.stream(self.prepareStream):
            if self.dynamicScaledOpticalFlow:
                from ..utils.SSIM import SSIM
                self.CompareNet = SSIM()
                print("Dynamic Scaled Optical Flow Enabled")
                if self.backend == "tensorrt":
                    print("Dynamic Scaled Optical Flow does not work with TensorRT, disabling", file=sys.stderr)
                    self.CompareNet = None
                if self.UHDMode:
                    print("Dynamic Scaled Optical Flow does not work with UHD Mode, disabling", file=sys.stderr)
                    self.CompareNet = None
            from .InterpolateArchs.GMFSS.GMFSS import GMFSS

            _pad = 64
            if self.dynamicScaledOpticalFlow:
                tmp = max(_pad, int(_pad / 0.25))
            else:
                tmp = max(_pad, int(_pad / self.scale))
            self.pw = math.ceil(self.width / tmp) * tmp
            self.ph = math.ceil(self.height / tmp) * tmp
            self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
            # caching the timestep tensor in a dict with the timestep as a float for the key
            self.timestepDict = {}
            for n in range(self.ceilInterpolateFactor):
                timestep = n / (self.ceilInterpolateFactor)
                timestep_tens = torch.tensor(
                    [timestep], dtype=self.dtype, device=self.device
                ).to(non_blocking=True)
                self.timestepDict[timestep] = timestep_tens
            self.flownet = GMFSS(
                model_path=self.interpolateModel,
                scale=self.scale,
                width=self.width,
                height=self.height,
                ensemble=self.ensemble,
            )
            
            self.flownet.eval().to(device=self.device, dtype=self.dtype)
            log("GMFSS loaded")
            log("Scale: " + str(self.scale))
            log("Using System CUDA: " + str(HAS_SYSTEM_CUDA))
            if not HAS_SYSTEM_CUDA:
                print("WARNING: System CUDA not found, falling back to PyTorch softsplat. This will be a bit slower.",file=sys.stderr)
            if self.backend == "tensorrt":
                warnAndLog(
                    "TensorRT is not implemented for GMFSS yet, falling back to PyTorch"
                )
        self.prepareStream.synchronize()

    @torch.inference_mode()
    def __call__(self, img1, writeQueue:Queue, transition=False, upscaleModel:UpscalePytorch = None):
        with torch.cuda.stream(self.stream):

            if self.frame0 is None:
                self.frame0 = self.frame_to_tensor(img1)
                self.stream.synchronize()
                return
                
            frame1 = self.frame_to_tensor(img1)
            for n in range(self.ceilInterpolateFactor-1):
                if not transition:
                    timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                    while self.flownet is None:
                        sleep(1)
                    timestep = self.timestepDict[timestep]
                    output = self.flownet(self.frame0, frame1, timestep)
                    if upscaleModel is not None:
                        output = upscaleModel(upscaleModel.frame_to_tensor(self.tensor_to_frame(output)))
                    else:
                        output = self.tensor_to_frame(output)
                    writeQueue.put(output)
                else:
                    if upscaleModel is not None:
                        img1 = upscaleModel(frame1[:, :, : self.height, : self.width])
                    writeQueue.put(img1)
            
            self.copyTensor(self.frame0, frame1)
           
        self.stream.synchronize()


class InterpolateRifeTorch(BaseInterpolate):
    @torch.inference_mode()
    def __init__(
        self,
        modelPath: str,
        ceilInterpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        device: str = "default",
        dtype: str = "auto",
        backend: str = "pytorch",
        UHDMode: bool = False,
        ensemble: bool = False,
        dynamicScaledOpticalFlow: bool = False,
        # trt options
        trt_optimization_level: int = 5,
        *args,
        **kwargs,
    ):
        if device == "default":
            if torch.cuda.is_available():
                device = torch.device(
                    "cuda", 0
                )  # 0 is the device index, may have to change later
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)

        printAndLog("Using device: " + str(device))

        self.interpolateModel = modelPath
        self.width = width
        self.height = height

        self.device = device
        self.dtype = self.handlePrecision(dtype)
        self.backend = backend
        self.ceilInterpolateFactor = ceilInterpolateFactor
        self.dynamicScaledOpticalFlow = dynamicScaledOpticalFlow
        self.CompareNet = None
        self.frame0 = None
        self.encode0 = None
        # set up streams for async processing
        self.scale = 1
        self.doEncodingOnFrame = True
        self.ensemble = ensemble

        self.trt_optimization_level = trt_optimization_level
        self.trt_cache_dir = os.path.dirname(
                modelPath
            )  # use the model directory as the cache directory
        self.UHDMode = UHDMode
        if self.UHDMode:
            self.scale = 0.5
        self._load()

    @torch.inference_mode()
    def _load(self):
        self.stream = torch.cuda.Stream()
        self.prepareStream = torch.cuda.Stream()
        with torch.cuda.stream(self.prepareStream):
            state_dict = torch.load(
                self.interpolateModel,
                map_location=self.device,
                weights_only=True,
                mmap=True,
            )
            # detect what rife arch to use

            ad = ArchDetect(self.interpolateModel)
            interpolateArch = ad.getArchName()
            _pad = 32
            match interpolateArch.lower():
                case "rife46":
                    from .InterpolateArchs.RIFE.rife46IFNET import IFNet

                    self.doEncodingOnFrame = False
                case "rife47":
                    from .InterpolateArchs.RIFE.rife47IFNET import IFNet

                    num_ch_for_encode = 4
                    self.encode = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 16, 3, 2, 1),
                        torch.nn.ConvTranspose2d(16, 4, 4, 2, 1),
                    )
                case "rife413":
                    from .InterpolateArchs.RIFE.rife413IFNET import IFNet, Head

                    num_ch_for_encode = 8
                    self.encode = Head()
                case "rife420":
                    from .InterpolateArchs.RIFE.rife420IFNET import IFNet, Head

                    num_ch_for_encode = 8
                    self.encode = Head()
                case "rife421":
                    from .InterpolateArchs.RIFE.rife421IFNET import IFNet, Head

                    num_ch_for_encode = 8
                    self.encode = Head()
                case "rife422lite":
                    from .InterpolateArchs.RIFE.rife422_liteIFNET import IFNet, Head

                    self.encode = Head()
                    num_ch_for_encode = 4
                case "rife425":
                    from .InterpolateArchs.RIFE.rife425IFNET import IFNet, Head

                    _pad = 64
                    num_ch_for_encode = 4
                    self.encode = Head()

                case _:
                    errorAndLog("Invalid Interpolation Arch")

            # model unspecific setup
            if self.dynamicScaledOpticalFlow:
                tmp = max(_pad, int(_pad / 0.25)) # set pad to higher for better dynamic optical scale support
            else:
                tmp = max(_pad, int(_pad / self.scale))
            self.pw = math.ceil(self.width / tmp) * tmp
            self.ph = math.ceil(self.height / tmp) * tmp
            self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
            # caching the timestep tensor in a dict with the timestep as a float for the key

            self.timestepDict = {}
            for n in range(self.ceilInterpolateFactor):
                timestep = n / (self.ceilInterpolateFactor)
                timestep_tens = torch.full(
                    (1, 1, self.ph, self.pw),
                    timestep,
                    dtype=self.dtype,
                    device=self.device,
                ).to(non_blocking=True)
                self.timestepDict[timestep] = timestep_tens
            # rife specific setup
            self.set_rife_args()  # sets backwarp_tenGrid and tenFlow_div
            self.flownet = IFNet(
                scale=self.scale,
                ensemble=self.ensemble,
                dtype=self.dtype,
                device=self.device,
                width=self.width,
                height=self.height,
            )

            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
                if "module." in k
            }
            head_state_dict = {
                k.replace("encode.", ""): v
                for k, v in state_dict.items()
                if "encode." in k
            }
            if self.doEncodingOnFrame:
                self.encode.load_state_dict(state_dict=head_state_dict, strict=True)
                self.encode.eval().to(device=self.device, dtype=self.dtype)
            self.flownet.load_state_dict(state_dict=state_dict, strict=False)
            self.flownet.eval().to(device=self.device, dtype=self.dtype)
            
            if self.dynamicScaledOpticalFlow:
                from ..utils.SSIM import SSIM
                self.CompareNet = SSIM()
                print("Dynamic Scaled Optical Flow Enabled")
                if self.backend == "tensorrt":
                    print("Dynamic Scaled Optical Flow does not work with TensorRT, disabling", file=sys.stderr)
                    self.CompareNet = None
                
                if self.UHDMode:
                    print("Dynamic Scaled Optical Flow does not work with UHD Mode, disabling", file=sys.stderr)
                    self.CompareNet = None
            
            if self.backend == "tensorrt":
                from .TensorRTHandler import TorchTensorRTHandler

                trtHandler = TorchTensorRTHandler(
                    trt_optimization_level=self.trt_optimization_level,
                )

                base_trt_engine_path = os.path.join(
                    os.path.realpath(self.trt_cache_dir),
                    (
                        f"{os.path.basename(self.interpolateModel)}"
                        + f"_{self.width}x{self.height}"
                        + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                        + f"_scale-{self.scale}"
                        + f"_{torch.cuda.get_device_name(self.device)}"
                        + f"_trt-{trtHandler.tensorrt_version}"
                        + f"_ensemble-{self.ensemble}"
                        + f"_torch_tensorrt-{trtHandler.torch_tensorrt_version}"
                        + (
                            f"_level-{self.trt_optimization_level}"
                            if self.trt_optimization_level is not None
                            else ""
                        )
                    ),
                )
                trt_engine_path = base_trt_engine_path + ".dyn"
                encode_trt_engine_path = base_trt_engine_path + "_encode.dyn"

                # lay out inputs
                # load flow engine
                if not os.path.isfile(trt_engine_path):
                    if not self.doEncodingOnFrame:
                        exampleInput = [
                            torch.zeros(
                                [1, 3, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                [1, 3, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                [1, 1, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros([2], dtype=torch.float, device=self.device),
                            torch.zeros(
                                [1, 2, self.ph, self.pw],
                                dtype=torch.float,
                                device=self.device,
                            ),
                        ]

                    else:
                        # if rife46
                        exampleInput = [
                            torch.zeros(
                                [1, 3, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                [1, 3, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                [1, 1, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros([2], dtype=torch.float, device=self.device),
                            torch.zeros(
                                [1, 2, self.ph, self.pw],
                                dtype=torch.float,
                                device=self.device,
                            ),
                            torch.zeros(
                                (1, num_ch_for_encode, self.ph, self.pw),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                (1, num_ch_for_encode, self.ph, self.pw),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                        ]

                        if not os.path.isfile(encode_trt_engine_path):
                            # build encode engine

                            encodedExampleInputs = [
                                torch.zeros(
                                    (1, 3, self.ph, self.pw),
                                    dtype=self.dtype,
                                    device=self.device,
                                ),
                            ]
                            trtHandler.build_engine(
                                model=self.encode,
                                dtype=self.dtype,
                                example_inputs=encodedExampleInputs,
                                device=self.device,
                                trt_engine_path=encode_trt_engine_path,
                            )

                        self.encode = trtHandler.load_engine(encode_trt_engine_path)

                    trtHandler.build_engine(
                        model=self.flownet,
                        dtype=self.dtype,
                        example_inputs=exampleInput,
                        device=self.device,
                        trt_engine_path=trt_engine_path,
                    )

                self.flownet = trtHandler.load_engine(trt_engine_path)
        torch.cuda.empty_cache()
        self.prepareStream.synchronize()

    @torch.inference_mode()
    def set_rife_args(self):
        self.tenFlow_div = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=torch.float32,
            device=self.device,
        )
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=torch.float32, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        ).to(dtype=torch.float32, device=self.device)
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=torch.float32, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        ).to(dtype=torch.float32, device=self.device)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    @torch.inference_mode()
    def __call__(self, img1, writeQueue:Queue, transition=False, upscaleModel:UpscalePytorch = None):
        with torch.cuda.stream(self.stream):

            if self.frame0 is None:
                self.frame0 = self.frame_to_tensor(img1)
                if self.doEncodingOnFrame:
                    self.encode0 = self.encode_Frame(self.frame0)
                self.stream.synchronize()
                return
                
            frame1 = self.frame_to_tensor(img1)
            if self.doEncodingOnFrame:
                encode1 = self.encode_Frame(frame1)
            
            closest_value = self.dynamicScaleCalculation(frame1)
            for n in range(self.ceilInterpolateFactor-1):
                if not transition:
                    timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                    while self.flownet is None:
                        sleep(1)
                    timestep = self.timestepDict[timestep]
                    if self.doEncodingOnFrame:
                        output = self.flownet(
                            self.frame0,
                            frame1,
                            timestep,
                            self.tenFlow_div,
                            self.backwarp_tenGrid,
                            self.encode0,
                            encode1,
                            closest_value
                        )
                    else:
                        output = self.flownet(
                            self.frame0, frame1, timestep, self.tenFlow_div, self.backwarp_tenGrid, closest_value
                        )
                    if upscaleModel is not None:
                        output = upscaleModel(upscaleModel.frame_to_tensor(self.tensor_to_frame(output)))
                    else:
                        output = self.tensor_to_frame(output)
                    writeQueue.put(output)
                else:
                    if upscaleModel is not None:
                        img1 = upscaleModel(frame1[:, :, : self.height, : self.width])
                    writeQueue.put(img1)
            
            self.copyTensor(self.frame0, frame1)
            if self.doEncodingOnFrame:
                self.copyTensor(self.encode0, encode1)

        self.stream.synchronize()

    @torch.inference_mode()
    def encode_Frame(self, frame: torch.Tensor):
        while self.encode is None:
            sleep(1)
        with torch.cuda.stream(self.prepareStream):
            frame = self.encode(frame)
        self.prepareStream.synchronize()
        return frame

class InterpolateRifeTensorRT(InterpolateRifeTorch):
    @torch.inference_mode()
    def __call__(self, img1, writeQueue:Queue, transition=False, upscaleModel:UpscalePytorch = None):
        with torch.cuda.stream(self.stream):

            if self.frame0 is None:
                self.frame0 = self.frame_to_tensor(img1)
                if self.doEncodingOnFrame:
                    self.encode0 = self.encode_Frame(self.frame0)
                self.stream.synchronize()
                return
                
            frame1 = self.frame_to_tensor(img1)
            if self.doEncodingOnFrame:
                encode1 = self.encode_Frame(frame1)
            
            for n in range(self.ceilInterpolateFactor-1):

                while self.flownet is None:
                    sleep(1)

                if not transition:
                    timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                    timestep = self.timestepDict[timestep]

                    if self.doEncodingOnFrame:
                        output = self.flownet(
                            self.frame0,
                            frame1,
                            timestep,
                            self.tenFlow_div,
                            self.backwarp_tenGrid,
                            self.encode0,
                            encode1,
                        )
                    else:
                        output = self.flownet(
                            self.frame0, frame1, timestep, self.tenFlow_div, self.backwarp_tenGrid
                        )

                    if upscaleModel is not None:
                        output = upscaleModel(upscaleModel.frame_to_tensor(self.tensor_to_frame(output)))
                    else:
                        output = self.tensor_to_frame(output)

                    writeQueue.put(output)

                else:
                    if upscaleModel is not None:
                        img1 = upscaleModel(frame1[:, :, : self.height, : self.width])
                    writeQueue.put(img1)
            
            self.copyTensor(self.frame0, frame1)
            if self.doEncodingOnFrame:
                self.copyTensor(self.encode0, encode1)

        self.stream.synchronize()


class InterpolateFactory:
    @staticmethod
    def build_interpolation_method(interpolate_model_path,backend):
        ad = ArchDetect(interpolate_model_path)
        base_arch = ad.getArchBase()
        match base_arch:
            case "rife":
                if backend == "tensorrt":
                    return InterpolateRifeTensorRT
                return InterpolateRifeTorch
            case "gmfss":
                return InterpolateGMFSSTorch
            case "gimm":
                return InterpolateGIMMTorch
