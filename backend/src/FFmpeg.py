import cv2
import os
import subprocess
import queue
import time
import sys
from multiprocessing import shared_memory
import mmap
import struct
import numpy as np
from .Util import currentDirectory, printAndLog

sysout = sys.stdout


class FFMpegRender:
    def __init__(
        self,
        inputFile: str,
        outputFile: str,
        interpolateFactor: int = 1,
        upscaleTimes: int = 1,
        encoder: str = "libx264",
        pixelFormat: str = "yuv420p",
        benchmark: bool = False,
        overwrite: bool = False,
        frameSetupFunction=None,
        crf: str = "18",
        sharedMemoryID: str = None,
        shm: shared_memory.SharedMemory = None,
    ):
        """
        Generates FFmpeg I/O commands to be used with VideoIO
        Options:
        inputFile: str, The path to the input file.
        outputFile: str, The path to the output file.
        interpolateTimes: int, this sets the multiplier for the framerate when interpolating, when only upscaling this will be set to 1.
        upscaleTimes: int,
        encoder: str, The exact name of the encoder ffmpeg will use (default=libx264)
        pixelFormat: str, The pixel format ffmpeg will use, (default=yuv420p)
        overwrite: bool, overwrite existing output file if it exists
        """
        self.inputFile = inputFile
        self.outputFile = outputFile

        # upsacletimes will be set to the scale of the loaded model with spandrel
        self.upscaleTimes = upscaleTimes
        self.interpolateFactor = interpolateFactor
        self.encoder = encoder
        self.pixelFormat = pixelFormat
        self.benchmark = benchmark
        self.overwrite = overwrite
        self.readingDone = False
        self.writingDone = False
        self.writeOutPipe = False
        self.previewFrame = None
        self.crf = crf
        self.frameSetupFunction = frameSetupFunction
        self.sharedMemoryID = sharedMemoryID
        self.shm = shm

        self.writeOutPipe = self.outputFile == "PIPE"

        self.readQueue = queue.Queue(maxsize=50)
        self.writeQueue = queue.Queue(maxsize=50)

    def getVideoProperties(self, inputFile: str = None):
        printAndLog("Getting Video Properties...")
        if inputFile is None:
            cap = cv2.VideoCapture(self.inputFile)
        else:
            cap = cv2.VideoCapture(inputFile)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        self.frameChunkSize = self.width * self.height * 3

    def getFFmpegReadCommand(self):
        printAndLog("Generating FFmpeg READ command...")
        command = [
            f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
            "-i",
            f"{self.inputFile}",
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
        return command

    def getFFmpegWriteCommand(self):
        printAndLog("Generating FFmpeg WRITE command...")
        if not self.outputFile == "PIPE":
            if not self.benchmark:
                # maybe i can split this so i can just use ffmpeg normally like with vspipe
                command = [
                    f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{self.width * self.upscaleTimes}x{self.height * self.upscaleTimes}",
                    "-r",
                    f"{self.fps * self.interpolateFactor}",
                    "-i",
                    "-",
                    "-i",
                    f"{self.inputFile}",
                    f"-crf",
                    f"{self.crf}",
                    "-pix_fmt",
                    self.pixelFormat,
                    "-c:a",
                    "copy",
                    f"{self.outputFile}",
                ]
                for i in self.encoder.split():
                    command.append(i)
            else:
                printAndLog("Using benchmark mode")
                command = [
                    f"{os.path.join(currentDirectory(),'bin','ffmpeg')}",
                    "-y",
                    "-v",
                    "warning",
                    "-stats",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-s",
                    f"{self.width * self.upscaleTimes}x{self.height * self.upscaleTimes}",
                    "-pix_fmt",
                    f"yuv420p",
                    "-r",
                    f"{self.fps * self.interpolateFactor}",
                    "-i",
                    "-",
                    "-benchmark",
                    "-f",
                    "null",
                    "-",
                ]
            if self.overwrite:
                command.append("-y")
            return command

    def readinVideoFrames(self):
        printAndLog("Starting Video Read")
        self.readProcess = subprocess.Popen(
            self.getFFmpegReadCommand(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        for i in range(self.totalFrames - 1):
            chunk = self.readProcess.stdout.read(self.frameChunkSize)
            chunk = self.frameSetupFunction(chunk)
            self.readQueue.put(chunk)
        printAndLog("Ending Video Read")
        self.readQueue.put(None)
        self.readingDone = True
        self.readProcess.stdout.close()
        self.readProcess.terminate()

    def returnFrame(self, frame):
        return frame

    def writeOutToSharedMemory(self):
        # Create a shared memory block

        buffer = self.shm.buf

        printAndLog(f"Shared memory name: {self.shm.name}")
        while True:
            if self.writingDone == True:
                self.shm.close()
                self.shm.unlink()
                break
            if self.previewFrame is not None:
                buffer[: self.frameChunkSize] = bytes(self.previewFrame)
                # Update the shared array
            time.sleep(0.1)

    def writeOutVideoFrames(self):
        """
        Writes out frames either to ffmpeg or to pipe
        This is determined by the --output command, which if the PIPE parameter is set, it outputs the chunk to pipe.
        A command like this is required,
        ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -framerate 24 -i - -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a copy out.mp4
        """
        startTime = time.time()
        print("Starting Write Out")
        if self.writeOutPipe == False:
            self.writeProcess = subprocess.Popen(
                self.getFFmpegWriteCommand(),
                stdin=subprocess.PIPE,
                text=True,
                universal_newlines=True,
            )
            while True:
                frame = self.writeQueue.get()

                if frame is None:
                    break
                self.writeProcess.stdin.buffer.write(frame)
                self.previewFrame = frame
            self.writeProcess.stdin.close()
            self.writeProcess.wait()

        else:
            sys.stdout = sysout
            while True:
                frame = self.writeQueue.get()
                if frame is None:
                    break

                sys.stdout.buffer.write(frame)
            sys.stdout.close()

        renderTime = time.time() - startTime
        self.writingDone = True
        printAndLog(
            f"Completed Write!\nTime to complete render: {round(renderTime, 2)}"
        )
