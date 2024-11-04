import torch
import torch.nn.functional as F

from .InterpolateArchs.DetectInterpolateArch import ArchDetect
import math
import os
import logging
import gc
from .Util import (
    printAndLog,
    errorAndLog,
    modelsDirectory,
    check_bfloat16_support,
    log,
    warnAndLog,
)
from time import sleep

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)


class InterpolateRifeTorch:
    """InterpolateRifeTorch class for video interpolation using RIFE model in PyTorch.

    Args:
        interpolateModelPath (str): Path to the pre-trained interpolation model.
        interpolateArch (str, optional): Interpolation architecture to use. Defaults to "rife413".
        width (int, optional): Width of the input frames. Defaults to 1920.
        height (int, optional): Height of the input frames. Defaults to 1080.
        device (str, optional): Device to use for computation. Defaults to "default".
        dtype (str, optional): Data type to use for computation. Defaults to "auto".
        backend (str, optional): Backend to use for computation. Defaults to "pytorch".
        UHDMode (bool, optional): Flag to enable UHD mode. Defaults to False.
        ensemble (bool, optional): Flag to enable ensemble mode. Defaults to False.
        trt_workspace_size (int, optional): Workspace size for TensorRT optimization. Defaults to 0.
        trt_max_aux_streams (int | None, optional): Maximum auxiliary streams for TensorRT optimization. Defaults to None.
        trt_optimization_level (int, optional): Optimization level for TensorRT optimization. Defaults to 5.
        trt_cache_dir (str, optional): Directory to cache TensorRT engine files. Defaults to modelsDirectory().
        trt_debug (bool, optional): Flag to enable TensorRT debug mode. Defaults to False.

    Methods:
        process(img0, img1, timestep):
            Processes the input frames and returns the interpolated frame.

        tensor_to_frame(frame):
            Converts a tensor to a frame for rendering.

        frame_to_tensor(frame):
            Converts a frame to a tensor for processing.
    def __init__(self, interpolateModelPath, interpolateArch="rife413", width=1920, height=1080, device="default", dtype="auto", backend="pytorch", UHDMode=False, ensemble=False, trt_workspace_size=0, trt_max_aux_streams=None, trt_optimization_level=5, trt_cache_dir=modelsDirectory(), trt_debug=False):
        pass

        Processes the input frames and returns the interpolated frame.

        Args:
            img0 (torch.Tensor): First input frame.
            img1 (torch.Tensor): Second input frame.
            timestep (float): Timestep between the input frames.

        Returns:
            torch.Tensor: Interpolated frame.
        pass

    def tensor_to_frame(self, frame):
        Converts a tensor to a frame for rendering.

        Args:
            frame (torch.Tensor): Input tensor representing a frame.

        Returns:
            numpy.ndarray: Rendered frame.
        pass

    def frame_to_tensor(self, frame):
        Converts a frame to a tensor for processing.

        Args:
            frame (numpy.ndarray): Input frame.

        Returns:
            torch.Tensor: Tensor representing the frame.
        pass"""

    @torch.inference_mode()
    def __init__(
        self,
        interpolateModelPath: str,
        ceilInterpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        device: str = "default",
        dtype: str = "auto",
        backend: str = "pytorch",
        UHDMode: bool = False,
        # trt options
        trt_workspace_size: int = 0,
        trt_max_aux_streams: int | None = None,
        trt_optimization_level: int = 5,
        trt_cache_dir: str = modelsDirectory(),
        trt_debug: bool = False,
        rife_trt_mode: str = "accurate",
        trt_static_shape: bool = True,
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

        self.interpolateModel = interpolateModelPath
        self.width = width
        self.height = height

        self.device = device
        self.dtype = self.handlePrecision(dtype)
        self.trt_workspace_size = trt_workspace_size
        self.trt_max_aux_streams = trt_max_aux_streams
        self.trt_optimization_level = trt_optimization_level
        self.trt_cache_dir = trt_cache_dir
        self.backend = backend
        self.ceilInterpolateFactor = ceilInterpolateFactor
        # set up streams for async processing
        self.scale = 1
        self.img0 = None
        self.f0encode = None
        self.rife46 = False
        self.gmfss = False
        self.trt_debug = trt_debug  # too much output, i would like a progress bar tho
        self.rife_trt_mode = rife_trt_mode
        self.trt_static_shape = trt_static_shape

        if UHDMode:
            self.scale = 0.5
        self._load()

    @torch.inference_mode()
    def _load(self):
        IFNet = None
        GMFSS = None
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
            interpolateArch = ad.getArch()
            log("interp arch" + interpolateArch.lower())

            _pad = 32
            match interpolateArch.lower():
                case "gmfss":
                    from .InterpolateArchs.GMFSS.GMFSS import GMFSS

                    _pad = 64
                    self.gmfss = True
                    self.rife46 = True
                case "rife46":
                    from .InterpolateArchs.RIFE.rife46IFNET import IFNet

                    self.rife46 = True
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
            tmp = max(_pad, int(_pad / self.scale))
            self.pw = math.ceil(self.width / tmp) * tmp
            self.ph = math.ceil(self.height / tmp) * tmp
            self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
            # caching the timestep tensor in a dict with the timestep as a float for the key
            self.timestepDict = {}
            if GMFSS is not None:
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
                )
                # self.dtype = torch.float32
                # warnAndLog("GMFSS does not support float16, switching to float32")
                self.flownet.eval().to(device=self.device, dtype=self.dtype)
                if self.backend == "tensorrt":
                    warnAndLog(
                        "TensorRT is not implemented for GMFSS yet, falling back to PyTorch"
                    )

            elif IFNet is not None:
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
                    ensemble=False,
                    dtype=self.dtype,
                    device=self.device,
                    width=self.width,
                    height=self.height,
                    rife_trt_mode=self.rife_trt_mode,
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
                if not self.rife46:
                    self.encode.load_state_dict(state_dict=head_state_dict, strict=True)
                    self.encode.eval().to(device=self.device, dtype=self.dtype)
                self.flownet.load_state_dict(state_dict=state_dict, strict=False)
                self.flownet.eval().to(device=self.device, dtype=self.dtype)
                if self.backend == "tensorrt":
                    import tensorrt
                    import torch_tensorrt
                    from .TensorRTHandler import TorchTensorRTHandler
                    trtHandler = TorchTensorRTHandler(trt_optimization_level=self.trt_optimization_level,trt_cache_dir=self.trt_cache_dir)

                    base_trt_engine_path = os.path.join(
                        os.path.realpath(self.trt_cache_dir),
                        (
                            f"{os.path.basename(self.interpolateModel)}"
                            + f"_{self.width}x{self.height}"
                            + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                            + f"_scale-{self.scale}"
                            + "_ensemble-False"
                            + f"_{torch.cuda.get_device_name(self.device)}"
                            + f"torch_tensorrt-{torch_tensorrt.__version__}"
                            + f"_trt-{tensorrt.__version__}"
                            + ("rife_trt_mode-" + self.rife_trt_mode)
                            + (
                                f"_workspace-{self.trt_workspace_size}"
                                if self.trt_workspace_size > 0
                                else ""
                            )
                            + (
                                f"_aux-{self.trt_max_aux_streams}"
                                if self.trt_max_aux_streams is not None
                                else ""
                            )
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
                        if self.rife46:
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
                            # if not rife46
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
                                printAndLog(
                                    "Building TensorRT engine {}".format(
                                        encode_trt_engine_path
                                    )
                                )
                                encodedExampleInputs = [
                                    torch.zeros(
                                        (1, 3, self.ph, self.pw),
                                        dtype=self.dtype,
                                        device=self.device,
                                    ),
                                ]
                                trtHandler.build_engine(model=self.encode,dtype=self.dtype,example_inputs=encodedExampleInputs,device=self.device,trt_engine_path=encode_trt_engine_path)

                            printAndLog(
                                f"Loading TensorRT engine from {encode_trt_engine_path}"
                            )
                            self.encode = trtHandler.load_engine(encode_trt_engine_path)

                        # export flow engine
                        printAndLog(
                            "Building TensorRT engine {}".format(trt_engine_path)
                        )
                        trtHandler.build_engine(model=self.flownet,dtype=self.dtype,example_inputs=exampleInput,device=self.device,trt_engine_path=trt_engine_path)
                        
                    printAndLog(f"Loading TensorRT engine from {trt_engine_path}")
                    self.flownet = trtHandler.load_engine(trt_engine_path)
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

    def handlePrecision(self, precision):
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
    def process(self, img0, img1, timestep, f0encode=None, f1encode=None):
        while self.flownet is None:
            sleep(1)
        with torch.cuda.stream(self.stream):
            timestep = self.timestepDict[timestep]
            if not self.gmfss:
                if not self.rife46:
                    output = self.flownet(
                        img0,
                        img1,
                        timestep,
                        self.tenFlow_div,
                        self.backwarp_tenGrid,
                        f0encode,
                        f1encode,
                    )
                else:
                    output = self.flownet(
                        img0, img1, timestep, self.tenFlow_div, self.backwarp_tenGrid
                    )
            else:
                # output = F.interpolate(self.flownet(img0, img1, timestep), (self.height, self.width), mode="bilinear")
                output = self.flownet(img0, img1, timestep)
        self.stream.synchronize()
        return self.tensor_to_frame(output)

    @torch.inference_mode()
    def uncacheFrame(self):
        self.f0encode = None
        self.img0 = None

    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor):
        return frame.float().byte().contiguous().cpu().numpy()

    @torch.inference_mode()
    def encode_Frame(self, frame: torch.Tensor):
        while self.encode is None:
            sleep(1)
        with torch.cuda.stream(self.prepareStream):
            frame = self.encode(frame)
        self.prepareStream.synchronize()
        return frame

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
