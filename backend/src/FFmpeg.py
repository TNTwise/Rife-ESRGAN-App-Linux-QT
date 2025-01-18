from multiprocessing import shared_memory
import sys
import time

from .utils.Util import log,  padFrame


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


class PauseManager:
    def __init__(self, paused_shared_memory_id):
        self.isPaused = False
        self.prevState = None
        self.paused_shared_memory_id = paused_shared_memory_id
        if self.paused_shared_memory_id is not None:
            self.pausedSharedMemory = shared_memory.SharedMemory(
                name=self.paused_shared_memory_id
            )

    def pause_manager(self):
        if self.paused_shared_memory_id is not None:
            self.isPaused = self.pausedSharedMemory.buf[0] == 1
            activate = self.prevState != self.isPaused
            self.prevState = self.isPaused
            return activate and self.isPaused


class InformationWriteOut:
    def __init__(
        self,
        sharedMemoryID,  # image memory id
        paused_shared_memory_id,
        outputWidth,
        outputHeight,
        croppedOutputWidth,
        croppedOutputHeight,
        totalOutputFrames,
        border_detect: bool = False,
    ):
        self.startTime = time.time()
        self.frameChunkSize = outputWidth * outputHeight * 3
        self.sharedMemoryID = sharedMemoryID
        self.paused_shared_memory_id = paused_shared_memory_id
        self.width = outputWidth
        self.height = outputHeight
        self.croppedOutputWidth = croppedOutputWidth
        self.croppedOututHeight = croppedOutputHeight
        self.totalOutputFrames = totalOutputFrames
        self.border_detect = border_detect
        self.previewFrame = None
        self.last_length = 0
        self.framesRendered = 1

        if self.sharedMemoryID is not None:
            self.shm = shared_memory.SharedMemory(
                name=self.sharedMemoryID, create=True, size=self.frameChunkSize
            )
        self.pausedManager = PauseManager(paused_shared_memory_id)
        self.isPaused = False
        self.stop = False

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

    def get_is_paused(self):
        return self.isPaused

    def calculateETA(self, framesRendered):
        """
        Calculates ETA

        Gets the time for every frame rendered by taking the
        elapsed time / completed iterations (files)
        remaining time = remaining iterations (files) * time per iteration

        """

        # Estimate the remaining time
        elapsed_time = time.time() - self.startTime
        time_per_iteration = elapsed_time / framesRendered
        remaining_iterations = self.totalOutputFrames - framesRendered
        remaining_time = remaining_iterations * time_per_iteration
        remaining_time = int(remaining_time)
        # convert to hours, minutes, and seconds
        hours, minutes, seconds = convertTime(remaining_time)
        return f"{hours}:{minutes}:{seconds}"

    def setPreviewFrame(self, frame):
        self.previewFrame = frame

    def setFramesRendered(self, framesRendered: int):
        self.framesRendered = framesRendered

    def stopWriting(self):
        self.stop = True

    def writeOutInformation(self, fcs):
        """
        fcs = framechunksize
        """
        # Create a shared memory block
        if self.sharedMemoryID is not None:
            log(f"Shared memory name: {self.shm.name}")
        
        while not self.stop:
            if self.previewFrame is not None and self.framesRendered > 0:
                
                # print out data to stdout
                fps = round(self.framesRendered / (time.time() - self.startTime))
                eta = self.calculateETA(framesRendered=self.framesRendered)
                message = f"FPS: {fps} Current Frame: {self.framesRendered} ETA: {eta}"
                self.realTimePrint(message)
                if self.sharedMemoryID is not None and self.previewFrame is not None:
                    # Update the shared array
                    if self.border_detect:
                        padded_frame = padFrame(
                            self.previewFrame,
                            self.width,
                            self.height,
                            self.croppedOutputWidth,
                            self.croppedOututHeight,
                            
                        )
                        self.shm.buf[:fcs] = bytes(padded_frame)
                    else:
                        
                        self.shm.buf[:fcs] = bytes(self.previewFrame)
                self.isPaused = self.pausedManager.pause_manager()
            time.sleep(0.1)
