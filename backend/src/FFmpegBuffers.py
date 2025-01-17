import queue
from abc import ABC, abstractmethod
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
from .utils.Encoders import Encoder, EncoderSettings


class Buffer(ABC):
    @abstractmethod
    def command(self) -> list[str]:
        pass


class FFmpegRead(Buffer):
    def __init__(self, inputFile, width, height, borderX, borderY):
        self.inputFile = inputFile
        self.width = width
        self.height = height
        self.borderX = borderX
        self.borderY = borderY
        self.inputFrameChunkSize = width * height * 3
        self.readProcess = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.readQueue = queue.Queue(maxsize=50)

    def command(self):
        log("Generating FFmpeg READ command...")
        command = [
            f"{FFMPEG_PATH}",
            "-i",
            f"{self.inputFile}",
            "-vf",
            f"crop={self.width}:{self.height}:{self.borderX}:{self.borderY}",
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

    def read_frame(self):
        chunk = self.readProcess.stdout.read(self.inputFrameChunkSize)
        if len(chunk) < self.inputFrameChunkSize:
            return None
        return chunk

    def read_frames_into_queue(self):
        while True:
            chunk = self.read_frame()
            if chunk is None:
                break
            self.readQueue.put(chunk)
        self.readQueue.put(None)

    def get_read_queue(self):
        return self.readQueue.get()

    def close(self):
        self.readProcess.stdout.close()
        self.readProcess.terminate()


class FFmpegWrite(Buffer):
    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        width: int,
        height: int,
        fps: float,
        crf: int,
        audio_bitrate: int,
        pixelFormat: str,
        overwrite: bool,
        custom_encoder: str,
        benchmark: bool,
        slowmo_mode: bool,
        upscaleTimes: int,
        ceilInterpolateFactor: int,
        video_encoder: EncoderSettings,
        audio_encoder: EncoderSettings,
    ):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.audio_bitrate = audio_bitrate
        self.pixelFormat = pixelFormat
        self.overwrite = overwrite
        self.custom_encoder = custom_encoder
        self.benchmark = benchmark
        self.slowmo_mode = slowmo_mode
        self.upscaleTimes = upscaleTimes
        self.ceilInterpolateFactor = ceilInterpolateFactor
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.writeQueue = queue.Queue(maxsize=50)
        self.previewFrame = None
        self.framesRendered: int = 1

    def command(self):
        log("Generating FFmpeg WRITE command...")
        if self.slowmo_mode:
            log("Slowmo mode enabled, will not merge audio or subtitles.")
        multiplier = (
            (self.fps * self.ceilInterpolateFactor)
            if not self.slowmo_mode
            else self.fps
        )
        if not self.benchmark:
            # maybe i can split this so i can just use ffmpeg normally like with vspipe
            command = [
                f"{FFMPEG_PATH}",
            ]

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

        log("FFMPEG WRITE COMMAND: " + str(command))
        return command

    def get_preview_frame(self):
        return self.previewFrame

    def get_num_frames_rendered(self):
        return self.framesRendered

    def put_frame_in_write_queue(self, frame):
        self.writeQueue.put(frame)

    def write_out_frames(self):
        """
        Writes out frames either to ffmpeg or to pipe
        This is determined by the --output command, which if the PIPE parameter is set, it outputs the chunk to pipe.
        A command like this is required,
        ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -framerate 24 -i - -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy out.mp4
        """
        log("Rendering")
        self.startTime = time.time()

        exit_code: int = 0
        try:
            with open(FFMPEG_LOG_FILE, "w") as f:
                with subprocess.Popen(
                    self.command(),
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
            print(str(e), file=sys.stderr)
            self.onErroredExit()
        if exit_code != 0:
            self.onErroredExit()

    def onErroredExit(self):
        self.writingDone = True
        print("FFmpeg failed to render the video.", file=sys.stderr)
        with open(FFMPEG_LOG_FILE, "r") as f:
            for line in f.readlines():
                print(line, file=sys.stderr)
        if self.video_encoder.getPresetTag() == "x264_vulkan":
            print("Vulkan encode failed, try restarting the render.", file=sys.stderr)
            print(
                "Make sure you have the latest drivers installed and your GPU supports vulkan encoding.",
                file=sys.stderr,
            )
        time.sleep(1)
        os._exit(1)