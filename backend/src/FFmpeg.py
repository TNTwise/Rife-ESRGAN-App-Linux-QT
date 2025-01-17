import cv2
import os
import subprocess
import queue
import sys
import time
import math
from .constants import FFMPEG_PATH, FFMPEG_LOG_FILE
from .utils.Util import (
    log,
    printAndLog,
)
from threading import Thread
import numpy as np
from .utils.Encoders import EncoderSettings


def convertTime(remaining_time):
    """
    Converts seconds to hours, minutes and seconds
    """
    hours = remaining_time // 3600
    remaining_time -= 3600 * hours
    minutes = remaining_time // 60
    remaining_time -= minutes * 60
    seconds = remaining_time
    if minutes < 10:
        minutes = str(f"0{minutes}")
    if seconds < 10:
        seconds = str(f"0{seconds}")
    return hours, minutes, seconds



class FFMpegRender:
    """Args:
        inputFile (str): The path to the input file.
        outputFile (str): The path to the output file.
        interpolateFactor (int, optional): Sets the multiplier for the framerate when interpolating. Defaults to 1.
        upscaleTimes (int, optional): Upscaling factor. Defaults to 1.
        encoder (str, optional): The exact name of the encoder ffmpeg will use. Defaults to "libx264".
        pixelFormat (str, optional): The pixel format ffmpeg will use. Defaults to "yuv420p".
        benchmark (bool, optional): Enable benchmark mode. Defaults to False.
        overwrite (bool, optional): Overwrite existing output file if it exists. Defaults to False.
        frameSetupFunction (function, optional): Function to setup frames. Defaults to None.
        crf (str, optional): Constant Rate Factor for video quality. Defaults to "18".
        sharedMemoryID (str, optional): ID for shared memory. Defaults to None.
        shm (shared_memory.SharedMemory, optional): Shared memory object. Defaults to None.
        inputFrameChunkSize (int, optional): Size of input frame chunks. Defaults to None.
        outputFrameChunkSize (int, optional): Size of output frame chunks. Defaults to None.
    pass
    Gets the properties of the video file.
    Args:
        inputFile (str, optional): The path to the input file. If None, uses the inputFile specified in the constructor. Defaults to None.
    pass
    Generates the FFmpeg command for reading video frames.
    Returns:
        list: The FFmpeg command for reading video frames.
    pass
    Generates the FFmpeg command for writing video frames.
    Returns:
        list: The FFmpeg command for writing video frames.
    pass
    Starts reading video frames using FFmpeg.
    pass
    Returns a frame.
    Args:
        frame: The frame to be returned.
    Returns:
        The returned frame.
    pass
    Prints data in real-time.
    Args:
        data: The data to be printed.
    pass
    Writes frames to shared memory.
    Args:
        fcs: The frame chunk size.
    pass
    Writes out video frames using FFmpeg.
    pass"""

    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        interpolateFactor: int = 1,
        upscaleTimes: int = 1,
        custom_encoder: str = None,
        crf: int = 18,
        video_encoder_preset: str = "libx264",
        audio_encoder_preset: str = "aac",
        audio_bitrate: str = "192k",
        pixelFormat: str = "yuv420p",
        benchmark: bool = False,
        overwrite: bool = False,
        sharedMemoryID: str = None,
        channels=3,
        upscale_output_resolution: str = None,
        slowmo_mode: bool = False,
        hdr_mode: bool = False,
        border_detect: bool = False,
        output_to_mpv: bool = False,
    ):
        """
        Generates FFmpeg I/O commands to be used with VideoIO
        Options:
        inputFile: str, The path to the input file.
        outputFile: str, The path to the output file.
        interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
        upscaleTimes: int,
        custom_encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
        pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)
        overwrite: bool, overwrite existing output file if it exists
        """
        self.inputFile = inputFile
        self.outputFile = outputFile

        # upsacletimes will be set to the scale of the loaded model with spandrel
        self.upscaleTimes = upscaleTimes
        self.interpolateFactor = interpolateFactor
        self.ceilInterpolateFactor = math.ceil(self.interpolateFactor)
        self.video_encoder_preset = video_encoder_preset
        if custom_encoder is None: # custom_encoder overrides these presets
            self.video_encoder = EncoderSettings(video_encoder_preset)
            self.audio_encoder = EncoderSettings(audio_encoder_preset)

        self.custom_encoder = custom_encoder
        self.pixelFormat = pixelFormat
        self.benchmark = benchmark
        self.overwrite = overwrite
        self.readingDone = False
        self.writingDone = False
        self.writeOutPipe = False
        self.previewFrame = None
        self.slowmo_mode = slowmo_mode
        self.crf = crf
        self.audio_bitrate = audio_bitrate
        self.sharedMemoryID = sharedMemoryID
        self.border_detect = border_detect
        self.upscale_output_resolution = upscale_output_resolution
        self.output_to_mpv = output_to_mpv

        self.subtitleFiles = []
        
        self.inputFrameChunkSize = self.width * self.height * channels
        self.outputFrameChunkSize = (
            self.width * self.upscaleTimes * self.height * self.upscaleTimes * channels
        )
        sharedMemoryChunkSize = (
            self.originalHeight
            * self.originalWidth
            * channels
            * self.upscaleTimes
            * self.upscaleTimes
        )
        self.sharedMemoryThread = Thread(
            target=lambda: self.writeOutInformation(sharedMemoryChunkSize)
        )

        self.totalOutputFrames = self.totalInputFrames * self.ceilInterpolateFactor

        self.writeOutPipe = self.outputFile == "PIPE"

        self.readQueue = queue.Queue(maxsize=50)
        self.writeQueue = queue.Queue(maxsize=50)

    def getVideoProperties(self, inputFile: str = None):
        log("Getting Video Properties...")
        if inputFile is None:
            cap = cv2.VideoCapture(self.inputFile)
        else:
            cap = cv2.VideoCapture(inputFile)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.originalWidth = self.width
        self.originalHeight = self.height
        self.borderX = 0
        self.borderY = 0 # set borders for cropping automatically to 0, will be overwritten if borders are detected 
        self.totalInputFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        self.outputFrameChunkSize = None

    def readinVideoFrames(self):
        log("Starting Video Read")

        log("Ending Video Read")
        self.readQueue.put(None)
        self.readingDone = True

    def returnFrame(self, frame):
        return frame

    def realTimePrint(self, data):
        data = str(data)
        # Clear the last line
        sys.stdout.write("\r" + " " * self.last_length)
        sys.stdout.flush()

        # Write the new line
        sys.stdout.write("\r" + data)
        sys.stdout.flush()

        # Update the length of the last printed line
        self.last_length = len(data)

    def calculateETA(self):
        """
        Calculates ETA

        Gets the time for every frame rendered by taking the
        elapsed time / completed iterations (files)
        remaining time = remaining iterations (files) * time per iteration

        """

        # Estimate the remaining time
        elapsed_time = time.time() - self.startTime
        time_per_iteration = elapsed_time / self.framesRendered
        remaining_iterations = self.totalOutputFrames - self.framesRendered
        remaining_time = remaining_iterations * time_per_iteration
        remaining_time = int(remaining_time)
        # convert to hours, minutes, and seconds
        hours, minutes, seconds = convertTime(remaining_time)
        return f"{hours}:{minutes}:{seconds}"

    def writeOutInformation(self, fcs):
        """
        fcs = framechunksize
        """
        # Create a shared memory block

        buffer = self.shm.buf

        log(f"Shared memory name: {self.shm.name}")
        while True:
            if self.writingDone:
                self.shm.close()
                self.shm.unlink()
                break
            if self.previewFrame is not None:
                # print out data to stdout
                fps = round(self.framesRendered / (time.time() - self.startTime))
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

            time.sleep(0.1)

    def onErroredExit(self):
        self.writingDone = True
        print("FFmpeg failed to render the video.",file=sys.stderr)
        with open(FFMPEG_LOG_FILE, "r") as f:
            for line in f.readlines():
                print(line,file=sys.stderr)
        if self.video_encoder_preset == 'x264_vulkan':
            print("Vulkan encode failed, try restarting the render.",file=sys.stderr)
            print("Make sure you have the latest drivers installed and your GPU supports vulkan encoding.",file=sys.stderr)
        time.sleep(1)
        os._exit(1)

    def padFrame(self, frame_bytes: bytes, target_width: int, target_height: int) -> bytes:
        R = 52
        G = 59
        B = 71
        """
        Pads the frame to the target resolution.
        
        Args:
            frame_bytes (bytes): The input frame in bytes.
            target_width (int): The target width for padding.
            target_height (int): The target height for padding.
        
        Returns:
            bytes: The padded frame in bytes.
        """
        # Convert bytes to numpy array
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_array = frame_array.reshape(
            (self.height * self.upscaleTimes, self.width * self.upscaleTimes, 3)
        )

        padded_frame = np.full(
            (target_height, target_width, 3), (R, G, B), dtype=np.uint8
        )

        # Calculate padding offsets
        y_offset = (target_height - self.height * self.upscaleTimes) // 2
        x_offset = (target_width - self.width * self.upscaleTimes) // 2

        # Place the original frame in the center of the padded frame
        padded_frame[
            y_offset : y_offset + self.height * self.upscaleTimes,
            x_offset : x_offset + self.width * self.upscaleTimes,
        ] = frame_array

        # Convert the padded frame back to bytes
        return padded_frame.tobytes()