from threading import Thread
import os
import math
from time import sleep

from .FFmpeg import FFMpegRender
from .utils.SceneDetect import SceneDetect
from .utils.Util import printAndLog, log, removeFile



class Render(FFMpegRender):
    """
    Subclass of FFmpegRender
    FFMpegRender options:
    inputFile: str, The path to the input file.
    outputFile: str, The path to the output file.
    interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
    encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
    pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)

    RenderOptions:
    interpolationMethod
    upscaleModel
    backend (pytorch,ncnn,tensorrt)
    device (cpu,cuda)
    precision (float16,float32)

    NOTE:
    Everything in here has to happen in a specific order:
    Get the video properties (res,fps,etc)
    set up upscaling/interpolation, this gets the scale for upscaling if upscaling is the current task
    assign framechunksize to a value, as this is needed to catch bytes and set up shared memory
    set up shared memory
    """

    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        # backend settings
        backend="pytorch",
        device="default",
        precision="float16",
        # model settings
        upscaleModel=None,
        interpolateModel=None,
        interpolateFactor: int = 1,
        tile_size=None,
        # ffmpeg settings
        encoder: str = "libx264",
        pixelFormat: str = "yuv420p",
        benchmark: bool = False,
        overwrite: bool = False,
        crf: str = "18",
        # misc
        pausedFile=None,
        sceneDetectMethod: str = "pyscenedetect",
        sceneDetectSensitivity: float = 3.0,
        sharedMemoryID: str = None,
        trt_optimization_level: int = 3,
        rife_trt_mode: str = "accurate",
        upscale_output_resolution: str = None,
    ):
        if pausedFile is None:
            pausedFile = os.path.basename(inputFile) + "_paused_state.txt"
        self.inputFile = inputFile
        self.pausedFile = pausedFile
        with open(self.pausedFile, "w") as f:
            f.write("False")
        self.backend = backend
        self.upscaleModel = upscaleModel
        self.interpolateModel = interpolateModel
        self.tilesize = tile_size
        self.device = device
        self.precision = precision
        self.upscaleTimes = 1  # if no upscaling, it will default to 1
        self.interpolateFactor = interpolateFactor
        # max timestep is a hack to make sure ncnn cache frames too early, and ncnn breaks if i modify the code at all so ig this is what we are doing
        self.maxTimestep = (interpolateFactor - 1) / interpolateFactor
        self.ncnn = self.backend == "ncnn"
        self.ceilInterpolateFactor = math.ceil(self.interpolateFactor)
        self.setupRender = self.returnFrame  # set it to not convert the bytes to array by default, and just pass chunk through
        self.setupFrame0 = None
        self.doEncodingOnFrame = False
        self.isPaused = False
        self.sceneDetectMethod = sceneDetectMethod
        self.sceneDetectSensitivty = sceneDetectSensitivity
        self.sharedMemoryID = sharedMemoryID
        self.trt_optimization_level = trt_optimization_level
        self.rife_trt_mode = rife_trt_mode
        self.uncacheNextFrame = False
        # get video properties early
        self.getVideoProperties(inputFile)

        printAndLog("Using backend: " + self.backend)
        if upscaleModel:
            self.setupUpscale()

            printAndLog("Using Upscaling Model: " + self.upscaleModel)
        if interpolateModel:
            self.setupInterpolate()

            printAndLog("Using Interpolation Model: " + self.interpolateModel)
        self.renderThread = Thread(target=self.render)
        super().__init__(
            inputFile=inputFile,
            outputFile=outputFile,
            interpolateFactor=interpolateFactor,
            upscaleTimes=self.upscaleTimes,
            encoder=encoder,
            pixelFormat=pixelFormat,
            benchmark=benchmark,
            overwrite=overwrite,
            crf=crf,
            sharedMemoryID=sharedMemoryID,
            channels=3,
            upscale_output_resolution=upscale_output_resolution,
        )

        self.sharedMemoryThread.start()
        self.readPausedFileThread1 = Thread(target=self.readPausedFileThread)
        self.ffmpegReadThread = Thread(target=self.readinVideoFrames)
        self.ffmpegWriteThread = Thread(target=self.writeOutVideoFrames)

        self.ffmpegReadThread.start()
        self.ffmpegWriteThread.start()
        self.renderThread.start()
        self.readPausedFileThread1.start()

    def readPausedFileThread(self):
        activate = True
        self.prevState = False
        while not self.writingDone:
            if os.path.isfile(self.pausedFile):
                with open(self.pausedFile, "r") as f:
                    self.isPaused = f.read().strip() == "True"
                    activate = self.prevState != self.isPaused
                if activate:
                    if self.isPaused:
                        self.hotUnload()
                        print("\nRender Paused")
                    else:
                        print("\nResuming Render")
                        self.hotReload()
                self.prevState = self.isPaused
            sleep(1)

    def i0Norm(self, frame):
        self.setupFrame0 = self.frameSetupFunction(frame)
        if self.doEncodingOnFrame:
            self.encodedFrame0 = self.encodeFrame(self.setupFrame0)

    def i1Norm(self, frame):
        self.setupFrame1 = self.frameSetupFunction(frame)
        if self.doEncodingOnFrame:
            self.encodedFrame1 = self.encodeFrame(self.setupFrame1)

    def onEndOfInterpolateCall(self):
        if self.ncnn:
            self.setupFrame1 = self.setupFrame0
        else:
            self.copyFrame(self.setupFrame0, self.setupFrame1)
            if self.doEncodingOnFrame:
                self.copyFrame(self.encodedFrame0, self.encodedFrame1)

    def renderInterpolate(self, frame, transition=False):
        if frame is not None:
            if self.setupFrame0 is None:
                self.i0Norm(frame)
                return
            self.i1Norm(frame)

            for n in range(self.ceilInterpolateFactor - 1):
                if not transition:
                    timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                    if self.doEncodingOnFrame:
                        frame = self.interpolate(
                            img0=self.setupFrame0,
                            img1=self.setupFrame1,
                            timestep=timestep,
                            f0encode=self.encodedFrame0,
                            f1encode=self.encodedFrame1,
                        )
                    else:
                        frame = self.interpolate(
                            img0=self.setupFrame0,
                            img1=self.setupFrame1,
                            timestep=timestep,
                        )
                elif self.ncnn:
                    self.interpolate(
                        img0=self.setupFrame0,
                        img1=self.setupFrame1,
                        timestep=self.maxTimestep,
                    )

                self.writeQueue.put(frame)

            self.onEndOfInterpolateCall()

    def render(self):
        while True:
            if not self.isPaused:
                frame = self.readQueue.get()
                if frame is None:
                    break
                if self.upscaleModel:
                    frame = self.upscale(self.frameSetupFunction(frame))

                if self.interpolateModel:
                    if self.sceneDetectMethod.lower() != "none":
                        self.renderInterpolate(frame, self.scDetectFunc(frame))
                    else:
                        self.renderInterpolate(frame, False)

                self.writeQueue.put(frame)
            else:
                sleep(1)
        self.writeQueue.put(None)
        removeFile(self.pausedFile)

    def setupUpscale(self):
        """
        This is called to setup an upscaling model if it exists.
        Maps the self.upscaleTimes to the actual scale of the model
        Maps the self.setupRender function that can setup frames to be rendered
        Maps the self.upscale the upscale function in the respective backend.
        For interpolation:
        Mapss the self.undoSetup to the tensor_to_frame function, which undoes the prep done in the FFMpeg thread. Used for SCDetect
        """
        printAndLog("Setting up Upscale")
        if self.backend == "pytorch" or self.backend == "tensorrt":
            from .pytorch.UpscaleTorch import UpscalePytorch
            upscalePytorch = UpscalePytorch(
                self.upscaleModel,
                device=self.device,
                precision=self.precision,
                width=self.width,
                height=self.height,
                backend=self.backend,
                tilesize=self.tilesize,
                trt_optimization_level=self.trt_optimization_level,
            )
            self.upscaleTimes = upscalePytorch.getScale()
            self.frameSetupFunction = upscalePytorch.bytesToFrame
            self.upscale = upscalePytorch.renderToNPArray
            self.hotUnload = upscalePytorch.hotUnload
            self.hotReload = upscalePytorch.hotReload

        if self.backend == "ncnn":
            from .ncnn.UpscaleNCNN import UpscaleNCNN, getNCNNScale
            path, last_folder = os.path.split(self.upscaleModel)

            self.upscaleModel = os.path.join(path, last_folder, last_folder)

            self.upscaleTimes = getNCNNScale(modelPath=self.upscaleModel)
            upscaleNCNN = UpscaleNCNN(
                modelPath=self.upscaleModel,
                num_threads=1,
                scale=self.upscaleTimes,
                gpuid=0,  # might have this be a setting
                width=self.width,
                height=self.height,
                tilesize=self.tilesize,
            )
            self.frameSetupFunction = self.returnFrame
            self.upscale = upscaleNCNN.Upscale
            self.hotUnload = upscaleNCNN.hotUnload
            self.hotReload = upscaleNCNN.hotReload
        if self.backend == "directml":
            from .onnx.UpscaleONNX import UpscaleONNX
            upscaleONNX = UpscaleONNX(
                modelPath=self.upscaleModel,
                precision=self.precision,
                width=self.width,
                height=self.height,
            )
            self.upscaleTimes = upscaleONNX.getScale()
            self.frameSetupFunction = upscaleONNX.bytesToFrame
            self.upscale = upscaleONNX.renderTensor

    def setupInterpolate(self):
        log("Setting up Interpolation")

        if self.sceneDetectMethod != "none":
            printAndLog("Scene Detection Enabled")

            scdetect = SceneDetect(
                sceneChangeMethod=self.sceneDetectMethod,
                sceneChangeSensitivity=self.sceneDetectSensitivty,
                width=self.width,
                height=self.height,
            )
            self.scDetectFunc = scdetect.detect

        else:
            printAndLog("Scene Detection Disabled")
            self.scDetectFunc = lambda x: False
        if self.backend == "ncnn":
            from .ncnn.InterpolateNCNN import InterpolateRIFENCNN
            interpolateRifeNCNN = InterpolateRIFENCNN(
                interpolateModelPath=self.interpolateModel,
                width=self.width,
                height=self.height,
                max_timestep=self.maxTimestep,
            )
            self.frameSetupFunction = interpolateRifeNCNN.normFrame
            self.undoSetup = interpolateRifeNCNN.uncacheFrame
            self.interpolate = interpolateRifeNCNN.process
            self.hotReload = interpolateRifeNCNN.hotReload
            self.hotUnload = interpolateRifeNCNN.hotUnload
            self.doEncodingOnFrame = False

        if self.backend == "pytorch" or self.backend == "tensorrt":
            from .pytorch.InterpolateTorch import InterpolateRifeTorch
            interpolateRifePytorch = InterpolateRifeTorch(
                modelPath=self.interpolateModel,
                ceilInterpolateFactor=self.ceilInterpolateFactor,
                width=self.width,
                height=self.height,
                device=self.device,
                dtype=self.precision,
                backend=self.backend,
                trt_optimization_level=self.trt_optimization_level,
            )
            self.frameSetupFunction = interpolateRifePytorch.frame_to_tensor
            self.undoSetup = interpolateRifePytorch.uncacheFrame
            self.interpolate = interpolateRifePytorch.process
            self.hotUnload = interpolateRifePytorch.hotUnload
            self.hotReload = interpolateRifePytorch.hotReload
            self.encodeFrame = interpolateRifePytorch.encode_Frame
            self.copyFrame = interpolateRifePytorch.copyTensor
            self.doEncodingOnFrame = not (interpolateRifePytorch.rife46)
