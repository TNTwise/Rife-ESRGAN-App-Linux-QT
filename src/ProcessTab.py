import subprocess
import os
from threading import Thread
import sys
import time
import numpy as np
import cv2
from multiprocessing import shared_memory
import time
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, Qt
from PySide6 import QtGui

from .Util import ffmpegPath, pythonPath, currentDirectory, modelsPath


class UpdateGUIThread(QThread):
    """
    Gets the latest bytes outputed from the shared memory and returns them in QImage format for display
    """
    latestPreviewPixmap = Signal(QtGui.QImage)

    def __init__(self, imagePreviewSharedMemoryID, outputVideoHeight,outputVideoWidth):
        super().__init__()
        self._stop_flag = False  # Boolean flag to control stopping
        self._mutex = QMutex()  # Atomic flag to control stopping
        self.imagePreviewSharedMemoryID = imagePreviewSharedMemoryID
        self.outputVideoHeight = outputVideoHeight
        self.outputVideoWidth = outputVideoWidth
    def run(self):
        while True:
            with QMutexLocker(self._mutex):
                if self._stop_flag:
                    break
            try:
                self.shm = shared_memory.SharedMemory(name=self.imagePreviewSharedMemoryID)
                image_bytes = self.shm.buf[:].tobytes()

                # Convert image bytes back to numpy array
                image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
                    (self.outputVideoHeight, self.outputVideoWidth, 3)
                )
                pixmap = self.convert_cv_qt(image_array)
                self.latestPreviewPixmap.emit(pixmap)
            except FileNotFoundError:
                #print("preview not available")
                pass
            time.sleep(0.1)
            
            
            
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv_img
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return convert_to_Qt_format

    def stop(self):
        with QMutexLocker(self._mutex):
            self._stop_flag = True
        self.shm.close()
        print("Closed Read Memory")

class ProcessTab:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.imagePreviewSharedMemoryID = "/image_preview"

        self.QButtonConnect()

    def QButtonConnect(self):
        # connect file select buttons
        self.parent.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.parent.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        # connect render button
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)

    def run(
        self,
        inputFile: str,
        outputPath: str,
        videoWidth: int,
        videoHeight: int,
        videoFps: float,
        videoFrameCount: int,
        upscaleTimes: int,
        interpolateTimes: int,
    ):
        self.inputFile = inputFile
        self.outputPath = outputPath
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.videoFps = videoFps
        self.videoFrameCount = videoFrameCount
        self.upscaleTimes = upscaleTimes
        self.interpolateTimes = interpolateTimes
        self.outputVideoWidth = videoWidth * upscaleTimes
        self.outputVideoHeight = videoHeight * upscaleTimes
        """
        Function to start the rendering process
        It will initially check for any issues with the current setup, (invalid file, no permissions, etc..)
        Then, based on the settings selected, it will build a command that is then passed into rve-backend
        Finally, It will handle the render via ffmpeg. Taking in the frames from pipe and handing them into ffmpeg on a sperate thread
        """

        # Gui changes
        self.parent.startRenderButton.setEnabled(False)

        # self.ffmpegWriteThread()
        writeThread = Thread(target=self.renderToPipeThread)
        writeThread.start()
        self.startGUIUpdate()

    def startGUIUpdate(self):
        
        self.workerThread = UpdateGUIThread(imagePreviewSharedMemoryID=self.imagePreviewSharedMemoryID,
                                            outputVideoHeight=self.outputVideoHeight,
                                            outputVideoWidth=self.outputVideoWidth
                                            )
        self.workerThread.latestPreviewPixmap.connect(self.updateProcessTab)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()

    def onRenderCompletion(self):
        self.workerThread.stop()
        self.workerThread.quit()
        self.workerThread.wait()
        

    def renderToPipeThread(self):
        command = [
            f"{pythonPath()}",
            os.path.join(currentDirectory(), "backend", "rve-backend.py"),
            "-i",
            self.inputFile,
            "-o",
            f"{self.outputPath}",
            "--interpolateModel",
            os.path.join(
                modelsPath(), "rife4.18.pkl"
            ),  # put actual model here, this is a placeholder
            "-b",
            "tensorrt",
            "--interpolateFactor",
            "2",
            "--shared_memory_id",
            f"{self.imagePreviewSharedMemoryID}",
        ]

        self.pipeInFrames = subprocess.run(
            command,
        )
        print("Done with render")
        self.onRenderCompletion()


    def updateProcessTab(self, qimage: QtGui.QImage):
        """
        Called by the worker QThread, and updates the GUI elements: Progressbar, Preview, FPS
        """
        try:
            p = qimage.scaled(600, 600, Qt.KeepAspectRatio)
            
            self.parent.previewLabel.setPixmap(QtGui.QPixmap.fromImage(p))
        except FileNotFoundError:
            # print("No preview yet!")
            pass
