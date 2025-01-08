import cv2
from abc import ABC
from dataclasses import dataclass
import os
import subprocess
import queue
import sys
import time
import math
from multiprocessing import shared_memory
from .constants import FFMPEG_PATH, FFMPEG_LOG_FILE
from .utils.Util import (
    log,
    printAndLog,
)
from threading import Thread
import numpy as np


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

class BorderDetect:
    def __init__(self, inputFile):
        self.inputFile = inputFile
    
    def processBorders(self):
        command = [
            f"{FFMPEG_PATH}",
            "-i",
            f"{self.inputFile}",
            "-vf",
            "cropdetect",
            "-f",
            "null",
            "-",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            universal_newlines=True,
        )
        output = process.communicate()
        return output
    
    def processOutput(self, output):
        borders = []
        for line in output[1].split('\n'):
            if "crop=" in line:
                crop_value = line.split("crop=")[1].split(' ')[0]
                borders.append(crop_value)
        
        if borders:
            def parse_crop(crop_str):
                # Expected format: "width:height:x:y"
                try:
                    width, height, x, y = map(int, crop_str.split(':'))
                    if width <= 0 or height <= 0:
                        log(f"Invalid crop dimensions: {crop_str}")
                        return None
                    return width, height, x, y
                except ValueError:
                    log(f"Invalid crop format: {crop_str}")
                    return None

            # Parse all crop values and filter out any invalid entries
            parsed_crops = [parse_crop(crop) for crop in borders]
            parsed_crops = [crop for crop in parsed_crops if crop is not None]

            if not parsed_crops:
                log("No valid crop values found.")
                return None

            # Determine the least cropped crop (i.e., largest area)
            least_cropped = max(parsed_crops, key=lambda dims: dims[0] * dims[1])
            least_cropped_str = f"{least_cropped[0]}:{least_cropped[1]}:{least_cropped[2]}:{least_cropped[3]}"

            return least_cropped_str

        return None
    
    def getBorders(self):
        output = self.processBorders()
        output = self.processOutput(output)
        width, height, borderX, borderY = map(int, output.split(':'))
        return width, height, borderX, borderY
   
@dataclass
class Encoder(ABC):
    preset_tag: str
    preInputsettings: str
    postInputSettings: str
    qualityControlMode: str = "-crf"

class copyAudio(Encoder):
    preset_tag = "copy_audio"
    preInputsettings = None
    postInputSettings = "-c:a copy"


class aac(Encoder):
    preset_tag = "aac"
    preInputsettings = None
    postInputSettings = "-c:a aac"


class libmp3lame(Encoder):
    preset_tag = "libmp3lame"
    preInputsettings = None
    postInputSettings = "-c:a libmp3lame"


class libx264(Encoder):
    preset_tag="libx264"
    preInputsettings = None
    postInputSettings = "-c:v libx264"

class libx265(Encoder):
    preset_tag="libx265"
    preInputsettings = None
    postInputSettings = "-c:v libx265"

class vp9(Encoder):
    preset_tag="vp9"
    preInputsettings = None
    postInputSettings = "-c:v libvpx-vp9"
    qualityControlMode: str = "-cq:v"

class av1(Encoder):
    preset_tag="av1"
    preInputsettings = None
    postInputSettings = "-c:v libsvtav1"

class x264_vulkan(Encoder):
    preset_tag="x264_vulkan"
    preInputsettings = "-init_hw_device vulkan=vkdev:0 -filter_hw_device vkdev"
    postInputSettings = '-filter:v format=nv12,hwupload -c:v h264_vulkan'
    # qualityControlMode: str = "-quality" # this is not implemented very well, quality ranges from 0-4 with little difference, so quality changing is disabled.

class x264_nvenc(Encoder):
    preset_tag="x264_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v h264_nvenc -preset slow"
    qualityControlMode: str = "-cq:v"

class x265_nvenc(Encoder):
    preset_tag="x265_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v hevc_nvenc -preset slow"
    qualityControlMode: str = "-cq:v"

class av1_nvenc(Encoder):
    preset_tag="av1_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v av1_nvenc -preset slow"
    qualityControlMode: str = "-cq:v"

class h264_vaapi(Encoder):
    preset_tag = "x264_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v h264_vaapi"
    qualityControlMode: str = "-qp"


class h265_vaapi(Encoder):
    preset_tag = "x265_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v hevc_vaapi"
    qualityControlMode: str = "-qp"


class av1_vaapi(Encoder):
    preset_tag = "av1_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v av1_vaapi"
    qualityControlMode: str = "-qp"


class EncoderSettings:
    def __init__(self, encoder_preset):
        self.encoder_preset = encoder_preset
        self.encoder:Encoder = self.getEncoder()
    
    def getEncoder(self) -> Encoder:
        for encoder in Encoder.__subclasses__():
            if encoder.preset_tag == self.encoder_preset:
                return encoder

    def getPreInputSettings(self) -> str:
        return self.encoder.preInputsettings

    def getPostInputSettings(self) -> str:
        return self.encoder.postInputSettings
    
    def getQualityControlMode(self) -> str:
        return self.encoder.qualityControlMode

    def getPresetTag(self) -> str:
        return self.encoder.preset_tag


   
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

        self.subtitleFiles = []
        
        self.inputFrameChunkSize = self.width * self.height * channels
        self.outputFrameChunkSize = (
            self.width * self.upscaleTimes * self.height * self.upscaleTimes * channels
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

        self.outputFrameChunkSize = None

    def getFFmpegReadCommand(self):
        log("Generating FFmpeg READ command...")
        command = [
            f"{FFMPEG_PATH}",
            "-i",
            f"{self.inputFile}",
            '-vf',
            f'crop={self.width}:{self.height}:{self.borderX}:{self.borderY}',
            "-f",
            "image2pipe",
            "-pix_fmt",
            "rgb24",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-",
        ]
        log("FFMPEG READ COMMAND: " + str(command))
        return command

    def getFFmpegWriteCommand(self):
        log("Generating FFmpeg WRITE command...")
        if self.slowmo_mode:
            log("Slowmo mode enabled, will not merge audio or subtitles.")
        multiplier = (self.fps * self.ceilInterpolateFactor) if not self.slowmo_mode else self.fps 
        if not self.benchmark:
            # maybe i can split this so i can just use ffmpeg normally like with vspipe
            command = [
                f"{FFMPEG_PATH}",]
            
            if self.custom_encoder is None:
                pre_in_set = self.video_encoder.getPreInputSettings()
                if pre_in_set is not None:
                    command += pre_in_set.split()


            command += [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.width * self.upscaleTimes}x{self.height * self.upscaleTimes}",
                "-r",
                f"{multiplier}",
                "-i",
                "-",
            ]

            if not self.slowmo_mode:
                command += [
                    "-i",
                    f"{self.inputFile}",
                    "-map",
                    "0:v",  # Map video stream from input 0
                    "-map",
                    "1:a?",  # Map all audio streams from input 1
                    "-map",
                    "1:s?",  # Map all subtitle streams from input 1
                ]
                command += self.audio_encoder.getPostInputSettings().split()
                if not self.audio_encoder.getPresetTag() == "copy_audio":
                    command += [
                        "-b:a",
                        self.audio_bitrate,
                    ]

            command += [
                "-pix_fmt",
                self.pixelFormat,
                "-c:s",
                "copy",
                "-loglevel",
                "error",
            ]

            if self.custom_encoder is not None:
                for i in self.custom_encoder.split():
                    command.append(i)
            else:
                command += self.video_encoder.getPostInputSettings().split()
                command += [self.video_encoder.getQualityControlMode(), str(self.crf)]

            command.append(
                f"{self.outputFile}",
            )

            if self.overwrite:
                command.append("-y")

        else:
            command = [
                f"{FFMPEG_PATH}",
                "-hide_banner",
                "-v",
                "warning",
                "-stats",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-video_size",
                f"{self.width*self.upscaleTimes}x{self.upscaleTimes*self.height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(multiplier),
                "-i",
                "-",
                "-benchmark",
                "-f",
                "null",
                "-",
            ]
        
        log("FFMPEG COMMAND: " + str(command))
        return command

    def readinVideoFrames(self):
        log("Starting Video Read")
        self.readProcess = subprocess.Popen(
            self.getFFmpegReadCommand(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        #import numpy as np
        #import cv2
        #frame_count = 0
        while True:
            chunk = self.readProcess.stdout.read(self.inputFrameChunkSize)
            if len(chunk) < self.inputFrameChunkSize:
                break
            self.readQueue.put(chunk)
            
            #np_array = np.frombuffer(chunk, np.uint8)
            #image = np_array.reshape((self.height, self.width, 3))
            #cv2.imwrite(f'frames/frame_{frame_count}.png', image)
            #frame_count += 1
        log("Ending Video Read")
        self.readQueue.put(None)
        self.readingDone = True
        self.readProcess.stdout.close()
        self.readProcess.terminate()

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

    def writeOutVideoFrames(self):
        """
        Writes out frames either to ffmpeg or to pipe
        This is determined by the --output command, which if the PIPE parameter is set, it outputs the chunk to pipe.
        A command like this is required,
        ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -framerate 24 -i - -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy out.mp4
        """
        log("Rendering")
        self.startTime = time.time()
        self.framesRendered: int = 1
        self.last_length: int = 0
        exit_code:int = 0
        try:
            with open (FFMPEG_LOG_FILE, "w") as f:  
                with subprocess.Popen(
                    self.getFFmpegWriteCommand(),
                    stdin=subprocess.PIPE,
                    stderr=f,
                    stdout=f,
                    text=True,
                    universal_newlines=True,
                ) as self.writeProcess:
                    while True:
                        frame = self.writeQueue.get()
                        if frame is None:
                            break
                        self.previewFrame = frame

                        self.writeProcess.stdin.buffer.write(frame)
                        self.framesRendered += 1

                    self.writeProcess.stdin.close()
                    self.writeProcess.wait()
                    exit_code = self.writeProcess.returncode

                    renderTime = time.time() - self.startTime
                    self.writingDone = True

                    printAndLog(f"\nTime to complete render: {round(renderTime, 2)}")
        except Exception as e:
            print(str(e),file=sys.stderr)
            self.onErroredExit()
        if exit_code != 0:
            self.onErroredExit()



    def padFrame(self, frame_bytes: bytes, target_width: int, target_height: int) -> bytes:
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

        padded_frame = np.full((target_height, target_width, 3), (52, 59, 71), dtype=np.uint8)

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
