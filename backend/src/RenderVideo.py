from threading import Thread
import os
import math
from time import sleep, time
import sys
from multiprocessing import shared_memory

from .FFmpeg import FFMpegRender, BorderDetect
from .utils.SceneDetect import SceneDetect
from .utils.Util import printAndLog, log


class Render(FFMpegRender):
    """
    Subclass of FFmpegRender
    FFMpegRender options:
    inputFile: str, The path to the input file.
    outputFile: str, The path to the output file.
    interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
    encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
    pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)

    interpolateOptions:
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
        pytorch_gpu_id: int = 0,
        ncnn_gpu_id: int = 0,
        # model settings
        upscaleModel=None,
        interpolateModel=None,
        interpolateFactor: int = 1,
        tile_size=None,
        # ffmpeg settings
        custom_encoder: str = "libx264",
        pixelFormat: str = "yuv420p",
        benchmark: bool = False,
        overwrite: bool = False,
        crf: str = "18",
        video_encoder_preset: str = "libx264",
        audio_encoder_preset: str = "aac",
        audio_bitrate: str = "192k",
        border_detect: bool = False,
        hdr_mode: bool = False,
        # misc
        pause_shared_memory_id=None,
        sceneDetectMethod: str = "pyscenedetect",
        sceneDetectSensitivity: float = 3.0,
        sharedMemoryID: str = None,
        trt_optimization_level: int = 3,
        upscale_output_resolution: str = None,
        UHD_mode: bool = False,
        slomo_mode: bool = False,
        dynamic_scaled_optical_flow: bool = False,
        ensemble: bool = False,
    ):
        
        
        self.inputFile = inputFile
        self.backend = backend
        self.upscaleModel = upscaleModel
        self.interpolateModel = interpolateModel
        self.tilesize = tile_size
        self.device = device
        self.precision = precision
        self.interpolateFactor = interpolateFactor
        # max timestep is a hack to make sure ncnn cache frames too early, and ncnn breaks if i modify the code at all so ig this is what we are doing
        # also used to help with performace and caching
        self.maxTimestep = (interpolateFactor - 1) / interpolateFactor
        self.ncnn = self.backend == "ncnn"
        self.ceilInterpolateFactor = math.ceil(self.interpolateFactor)
        self.setupRender = self.returnFrame  # set it to not convert the bytes to array by default, and just pass chunk through
        self.setupFrame0 = None
        self.interpolateOption = None
        self.upscaleOption = None
        self.isPaused = False
        self.sceneDetectMethod = sceneDetectMethod
        self.sceneDetectSensitivty = sceneDetectSensitivity
        self.sharedMemoryID = sharedMemoryID
        self.trt_optimization_level = trt_optimization_level
        self.uncacheNextFrame = False
        self.UHD_mode = UHD_mode
        self.dynamic_scaled_optical_flow = dynamic_scaled_optical_flow
        self.ensemble = ensemble
        self.pytorch_gpu_id = pytorch_gpu_id
        self.ncnn_gpu_id = ncnn_gpu_id
        
        
        # get video properties early
        self.getVideoProperties(inputFile)

        

        if border_detect:
            print("Detecting borders", file=sys.stderr)
            borderDetect = BorderDetect(inputFile=self.inputFile)
            self.width, self.height, self.borderX, self.borderY = borderDetect.getBorders()
            log(f"Detected borders: Width,Height:{self.width}x{self.height}, X,Y: {self.borderX}x{self.borderY}")


        printAndLog("Using backend: " + self.backend)
        # upscale has to be called first to get the scale of the upscale model
        if upscaleModel:
            self.setupUpscale()

            printAndLog("Using Upscaling Model: " + self.upscaleModel)
        else:
            self.upscaleTimes = 1  # if no upscaling, it will default to 1
        if interpolateModel:
            self.setupInterpolate()

            printAndLog("Using Interpolation Model: " + self.interpolateModel)

        
        # has to be after to detect upscale times
        sharedMemoryChunkSize = (
            self.originalHeight
            * self.originalWidth
            * 3 # channels
            * self.upscaleTimes
            * self.upscaleTimes
        )

        self.shm = shared_memory.SharedMemory(
            name=self.sharedMemoryID, create=True, size=sharedMemoryChunkSize
        )

        super().__init__(
            inputFile=inputFile,
            outputFile=outputFile,
            interpolateFactor=interpolateFactor,
            upscaleTimes=self.upscaleTimes,
            custom_encoder=custom_encoder,
            pixelFormat=pixelFormat,
            benchmark=benchmark,
            overwrite=overwrite,
            crf=crf,
            video_encoder_preset=video_encoder_preset,
            audio_encoder_preset=audio_encoder_preset,
            audio_bitrate=audio_bitrate,
            sharedMemoryID=sharedMemoryID,
            channels=3,
            upscale_output_resolution=upscale_output_resolution,
            slowmo_mode=slomo_mode,
            hdr_mode=hdr_mode,
            border_detect=border_detect,
        )
        
        
        self.renderThread = Thread(target=self.render)
        self.ffmpegReadThread = Thread(target=self.readinVideoFrames)
        self.ffmpegWriteThread = Thread(target=self.writeOutVideoFrames)
        self.sharedMemoryThread = Thread(
            target=lambda: self.writeOutInformation(sharedMemoryChunkSize, pause_shared_memory_id=pause_shared_memory_id) 
        )
        self.sharedMemoryThread.start()
        

        self.ffmpegReadThread.start()
        self.ffmpegWriteThread.start()
        self.renderThread.start()
        

    def writeOutInformation(self, fcs, pause_shared_memory_id=None):
        """
        fcs = framechunksize
        """
        # Create a shared memory block

        buffer = self.shm.buf
        activate = True
        self.prevState = False

        try:
            self.pausedSharedMemory = shared_memory.SharedMemory(name=pause_shared_memory_id)
        except Exception as e:
            self.pausedSharedMemory = shared_memory.SharedMemory(name=pause_shared_memory_id, create=True, size=1) # create it if it doesnt exist
            print("Error reading paused shared memory: " + str(e), sys.stderr)

        log(f"Shared memory name: {self.shm.name}")

        while True:

            if self.writingDone:
                self.shm.close()
                self.shm.unlink()
                break

            if self.previewFrame is not None:

                # print out data to stdout
                fps = round(self.framesRendered / (time() - self.startTime))
                eta = self.calculateETA()
                message = f"FPS: {fps} Current Frame: {self.framesRendered} ETA: {eta}"
                self.realTimePrint(message)
                if self.sharedMemoryID is not None and self.previewFrame is not None:

                    # Update the shared array
                    if self.border_detect:

                        padded_frame = self.padFrame(
                            self.previewFrame,
                            self.originalWidth * self.upscaleTimes,
                            self.originalHeight * self.upscaleTimes,
                        )
                        buffer[:fcs] = bytes(padded_frame)

                    else:

                        buffer[:fcs] = bytes(self.previewFrame)

            if pause_shared_memory_id is not None:

                self.isPaused = self.pausedSharedMemory.buf[0] == 1
                activate = self.prevState != self.isPaused
                if activate:
                    if self.isPaused:
                        if self.interpolateOption:
                            self.interpolateOption.hotUnload()
                        if self.upscaleOption:
                            self.upscaleOption.hotUnload()
                        print("\nRender Paused")
                    else:
                        print("\nResuming Render")
                        if self.upscaleOption:
                            self.upscaleOption.hotReload()
                        if self.interpolateOption:
                            self.interpolateOption.hotReload()
                self.prevState = self.isPaused
            sleep(0.1)

    

    def render(self):
        while True:
            if not self.isPaused:
                frame = self.readQueue.get()
                if frame is None:
                    break

                if self.interpolateModel:
                    self.interpolateOption(
                        img1=frame,
                        writeQueue=self.writeQueue,
                        transition=self.sceneDetect.detect(frame),
                        upscaleModel=self.upscaleOption,
                    )
                if self.upscaleModel:
                    frame = self.upscaleOption(
                        self.upscaleOption.frame_to_tensor(frame)
                    )

                self.writeQueue.put(frame)
            else:
                sleep(1)
        self.writeQueue.put(None)

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

            self.upscaleOption = UpscalePytorch(
                self.upscaleModel,
                device=self.device,
                precision=self.precision,
                width=self.width,
                height=self.height,
                backend=self.backend,
                tilesize=self.tilesize,
                trt_optimization_level=self.trt_optimization_level,
            )
            self.upscaleTimes = self.upscaleOption.getScale()

        if self.backend == "ncnn":
            from .ncnn.UpscaleNCNN import UpscaleNCNN, getNCNNScale

            path, last_folder = os.path.split(self.upscaleModel)
            self.upscaleModel = os.path.join(path, last_folder, last_folder)
            self.upscaleTimes = getNCNNScale(modelPath=self.upscaleModel)
            self.upscaleOption = UpscaleNCNN(
                modelPath=self.upscaleModel,
                num_threads=1,
                scale=self.upscaleTimes,
                gpuid=self.ncnn_gpu_id,  # might have this be a setting
                width=self.width,
                height=self.height,
                tilesize=self.tilesize,
            )

        if self.backend == "directml":  # i dont want to work with this shit
            from .onnx.UpscaleONNX import UpscaleONNX

            upscaleONNX = UpscaleONNX(
                modelPath=self.upscaleModel,
                precision=self.precision,
                width=self.width,
                height=self.height,
            )

    def setupInterpolate(self):
        log("Setting up Interpolation")
        self.sceneDetect = SceneDetect(
            sceneChangeMethod=self.sceneDetectMethod,
            sceneChangeSensitivity=self.sceneDetectSensitivty,
            width=self.width,
            height=self.height,
        )
        if self.sceneDetectMethod != "none":
            printAndLog("Scene Detection Enabled")

        else:
            printAndLog("Scene Detection Disabled")

        if self.backend == "ncnn":
            from .ncnn.InterpolateNCNN import InterpolateRIFENCNN

            self.interpolateOption = InterpolateRIFENCNN(
                interpolateModelPath=self.interpolateModel,
                width=self.width,
                height=self.height,
                gpuid=self.ncnn_gpu_id,
                max_timestep=self.maxTimestep,
                interpolateFactor=self.ceilInterpolateFactor,
            )

        if self.backend == "pytorch" or self.backend == "tensorrt":
            from .pytorch.InterpolateTorch import InterpolateFactory

            self.interpolateOption = InterpolateFactory.build_interpolation_method(
                self.interpolateModel,
                self.backend,
            )(
                modelPath=self.interpolateModel,
                ceilInterpolateFactor=self.ceilInterpolateFactor,
                width=self.width,
                height=self.height,
                device=self.device,
                dtype=self.precision,
                backend=self.backend,
                UHDMode=self.UHD_mode,
                trt_optimization_level=self.trt_optimization_level,
                ensemble=self.ensemble,
                dynamicScaledOpticalFlow=self.dynamic_scaled_optical_flow,
                max_timestep=self.maxTimestep,
            )
