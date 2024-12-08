import argparse
import os
import logging
from src.RenderVideo import Render

from src.utils.Util import (
    checkForPytorchCUDA,
    checkForPytorchROCM,
    checkForNCNN,
    checkForTensorRT,
    check_bfloat16_support,
    checkForDirectML,
    checkForDirectMLHalfPrecisionSupport,
    checkForGMFSS,
    get_pytorch_vram,
)


class HandleApplication:
    def __init__(self):
        self.args = self.handleArguments()
        if not self.args.list_backends:
            self.checkArguments()
            Render(
                # model settings
                inputFile=self.args.input,
                outputFile=self.args.output,
                interpolateModel=self.args.interpolate_model,
                interpolateFactor=self.args.interpolate_factor,
                upscaleModel=self.args.upscale_model,
                tile_size=self.args.tilesize,
                # backend settings
                device="default",
                backend=self.args.backend,
                precision=self.args.precision,
                # ffmpeg settings
                overwrite=self.args.overwrite,
                crf=self.args.crf,
                benchmark=self.args.benchmark,
                encoder=self.args.custom_encoder,
                # misc settingss
                pausedFile=self.args.paused_file,
                sceneDetectMethod=self.args.scene_detect_method,
                sceneDetectSensitivity=self.args.scene_detect_threshold,
                sharedMemoryID=self.args.shared_memory_id,
                trt_optimization_level=self.args.tensorrt_opt_profile,
                upscale_output_resolution=self.args.upscale_output_resolution,
                UHD_Mode=self.args.UHD_mode,
            )
        else:
            half_prec_supp = False
            availableBackends = []
            printMSG = ""

            if checkForTensorRT():
                """
                checks for tensorrt availability, and the current gpu works with it (if half precision is supported)
                Trt 10 only supports RTX 20 series and up.
                Half precision is only availaible on RTX 20 series and up
                """
                import torch

                half_prec_supp = check_bfloat16_support()
                if half_prec_supp:
                    import tensorrt

                    availableBackends.append("tensorrt")
                    printMSG += f"TensorRT Version: {tensorrt.__version__}\n"
                else:
                    printMSG += "ERROR: Cannot use tensorrt backend, as it is not supported on your current GPU"
            if checkForPytorchCUDA():
                import torch

                availableBackends.append("pytorch (cuda)")
                printMSG += f"PyTorch Version: {torch.__version__}\n"
                half_prec_supp = check_bfloat16_support()
            if checkForPytorchROCM():
                availableBackends.append("pytorch (rocm)")
                import torch

                printMSG += f"PyTorch Version: {torch.__version__}\n"
                half_prec_supp = check_bfloat16_support()
                
            if checkForNCNN():
                availableBackends.append("ncnn")
                printMSG += f"NCNN Version: 20220729\n"
                from rife_ncnn_vulkan_python import Rife
            if checkForDirectML():
                availableBackends.append("directml")
                import onnxruntime as ort

                printMSG += f"ONNXruntime Version: {ort.__version__}\n"
                half_prec_supp = checkForDirectMLHalfPrecisionSupport()
            printMSG += f"Half precision support: {half_prec_supp}\n"
           
            print("Available Backends: " + str(availableBackends))
            print(printMSG)

    def handleArguments(self) -> argparse.ArgumentParser:
        """_summary_

        Args:
            args (_type_): _description_

        """
        parser = argparse.ArgumentParser(
            description="Backend to RVE, used to upscale and interpolate videos"
        )

        parser.add_argument(
            "-i",
            "--input",
            default=None,
            help="input video path",
            type=str,
        )
        parser.add_argument(
            "-o",
            "--output",
            default=None,
            help="output video path or PIPE",
            type=str,
        )

        parser.add_argument(
            "-l",
            "--overlap",
            help="overlap size on tiled rendering (default=10)",
            default=0,
            type=int,
        )
        parser.add_argument(
            "-b",
            "--backend",
            help="backend used to upscale image. (pytorch/ncnn/tensorrt/directml, default=pytorch)",
            default="pytorch",
            type=str,
        )
        parser.add_argument(
            "--upscale_model",
            help="Direct path to upscaling model, will automatically upscale if model is valid.",
            type=str,
        )
        parser.add_argument(
            "--interpolate_model",
            help="Direct path to interpolation model, will automatically upscale if model is valid.\n(Downloadable Options: [rife46, rife47, rife415, rife418, rife420, rife422, rife422lite]))",
            type=str,
        )
        parser.add_argument(
            "--interpolate_factor",
            help="Multiplier for interpolation, will round up to nearest integer for interpolation but the fps will be correct",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--precision",
            help="sets precision for model, (auto/float16/float32, default=auto)",
            default="auto",
        )
        parser.add_argument(
            "--tensorrt_opt_profile",
            help="sets tensorrt optimization profile for model, (1/2/3/4/5, default=3)",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--scene_detect_method",
            help="Scene change detection to avoid interpolating transitions. (options=mean, mean_segmented, none)\nMean segmented splits up an image, and if an arbitrary number of segments changes are detected within the segments, it will trigger a scene change. (lower sensativity thresholds are not recommended)",
            type=str,
            default="pyscenedetect",
        )
        parser.add_argument(
            "--scene_detect_threshold",
            help="Scene change detection sensitivity, lower number means it has a higher chance of detecting scene changes, with risk of detecting too many.",
            type=float,
            default=4.0,
        )
        parser.add_argument(
            "--overwrite",
            help="Overwrite output video if it already exists.",
            action="store_true",
        )
        parser.add_argument(
            "--crf",
            help="Constant rate factor for videos, lower setting means higher quality.",
            default="18",
        )
        parser.add_argument(
            "--custom_encoder",
            help="custom encoder",
            default="-c:v libx264",
            type=str,
        )
        parser.add_argument(
            "--tilesize",
            help="upscale images in smaller chunks, default is the size of the input video",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--benchmark",
            help="Benchmark without saving video",
            action="store_true",
        )
        parser.add_argument(
            "--UHD_mode",
            help="Lowers the resoltion flow is calculated at, speeding up model and saving vram. Helpful for higher resultions.",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--shared_memory_id",
            help="Memory ID to share preview ons",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--list_backends",
            help="list out available backends",
            action="store_true",
        )
        parser.add_argument(
            "--paused_file",
            help="File to store paused state (True means paused, False means unpaused)",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--upscale_output_resolution",
            help="Resolution of output video, this is helpful for 4x models when you only want 2x upscaling. Ex: (1920x1080)",
            type=str,
            default=None,
        )

        return parser.parse_args()

    def fullModelPathandName(self):
        return os.path.join(self.args.modelPath, self.args.modelName)

    def checkArguments(self):
        if (
            os.path.isfile(self.args.output)
            and not self.args.overwrite
            and not self.args.benchmark
        ):
            raise os.error("Output file already exists!")
        if not os.path.isfile(self.args.input):
            raise os.error("Input file does not exist!")
        if self.args.tilesize < 0:
            raise ValueError("Tilesize must be greater than 0")
        if self.args.interpolate_factor < 0:
            raise ValueError("Interpolation factor must be greater than 0")
        if self.args.interpolate_factor == 1 and self.args.interpolate_model:
            raise ValueError(
                "Interpolation factor must be greater than 1 if interpolation model is used.\nPlease use --interpolateFactor 2 for 2x interpolation!"
            )
        if self.args.interpolate_factor != 1 and not self.args.interpolate_model:
            raise ValueError(
                "Interpolation factor must be 1 if no interpolation model is used.\nPlease use --interpolateFactor 1 for no interpolation!"
            )


if __name__ == "__main__":
    HandleApplication()
