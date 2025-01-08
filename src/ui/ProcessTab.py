import subprocess
import os
from threading import Thread
import re
import time
from multiprocessing import shared_memory

from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QColor
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QMessageBox

from .RenderQueue import RenderQueue

from .AnimationHandler import AnimationHandler
from .QTcustom import (
    UpdateGUIThread,
    show_layout_widgets,
    hide_layout_widgets,
)
from ..constants import (
    BACKEND_PATH,
    PYTHON_PATH,
    MODELS_PATH,
    CUSTOM_MODELS_PATH,
    IMAGE_SHARED_MEMORY_ID,
    PAUSED_STATE_SHARED_MEMORY_ID,
)
from ..Util import (
    log,
)
from ..DownloadModels import DownloadModel
from .SettingsTab import Settings
from ..DiscordRPC import DiscordRPC
from ..ModelHandler import (
    getModels
)
from .RenderQueue import RenderOptions

class ProcessTab:
    def __init__(self, parent, settings: Settings):
        self.parent = parent
        self.renderTextOutputList = None
        self.isOverwrite = False
        self.outputVideoHeight = None
        self.outputVideoWidth = None
        self.currentFrame = 0
        self.animationHandler = AnimationHandler()
        self.tileUpAnimationHandler = AnimationHandler()
        self.tileDownAnimationHandler = AnimationHandler()
        self.settings = settings
        self.qualityToCRF = {
            "Low": "28",
            "Medium": "23",
            "High": "18",
            "Very High": "15",
        }
        # encoder dict
        # key is the name in RVE gui
        # value is the encoder used

        # get default backend
        self.QConnect()
        self.populateModels(self.parent.backendComboBox.currentText())


    def populateModels(self, backend) -> dict:
        """
        returns
        the current models available given a method (interpolate, upscale) and a backend (ncnn, tensorrt, pytorch)
        """
        interpolateModels, upscaleModels = getModels(backend)
        self.parent.interpolateModelComboBox.clear()
        self.parent.upscaleModelComboBox.clear()
        self.parent.interpolateModelComboBox.addItems(
            ["None"] + list(interpolateModels.keys())
        )
        self.parent.upscaleModelComboBox.addItems(["None"] + list(upscaleModels.keys()))
       

    def onTilingSwitch(self):
        if self.parent.tilingCheckBox.isChecked():
            self.parent.tileSizeContainer.setVisible(True)
            self.tileDownAnimationHandler.dropDownAnimation(
                self.parent.tileSizeContainer
            )
        else:
            self.tileUpAnimationHandler.moveUpAnimation(self.parent.tileSizeContainer)
            self.parent.tileSizeContainer.setVisible(False)

    def QConnect(self):
        # connect file select buttons
        self.parent.addToRenderQueueButton.clicked.connect(self.parent.addToRenderQueue)
        self.parent.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.parent.inputFileText.textChanged.connect(self.parent.loadVideo)
        self.parent.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        # connect render button
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        # set tile size visible to false by default
        self.parent.tileSizeContainer.setVisible(False)
        #set slo mo container visable to false by default
        self.parent.interpolateContainer_2.setVisible(False)
        # connect up tilesize container visiable
        self.parent.tilingCheckBox.stateChanged.connect(self.onTilingSwitch)

        self.parent.interpolationMultiplierSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )

        self.parent.upscaleModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.interpolateModelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
        )

        self.parent.backendComboBox.currentIndexChanged.connect(
            lambda: self.populateModels(self.parent.backendComboBox.currentText())
        )
        # connect up pausing
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.pauseRenderButton.clicked.connect(self.pauseRender)
        self.parent.killRenderButton.clicked.connect(self.killRenderProcess)

    def killRenderProcess(self):
        try:  # kills  render process if necessary
            self.renderProcess.terminate()
        except AttributeError:
            log("No render process!")

    def pauseRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 1 # 1 = True
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(True)
        self.parent.startRenderButton.setEnabled(True)

    def resumeRender(self):
        shmbuf = self.pausedSharedMemory.buf
        shmbuf[0] = 0 # 0 = False
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.onRenderButtonsContiainer.setEnabled(True)
        self.parent.startRenderButton.setVisible(False)

    def startGUIUpdate(self):
        while self.outputVideoHeight is None:
            time.sleep(0.1)
        self.workerThread = UpdateGUIThread(
            parent=self,
            imagePreviewSharedMemoryID=IMAGE_SHARED_MEMORY_ID,
            outputVideoHeight=self.outputVideoHeight,
            outputVideoWidth=self.outputVideoWidth,
        )
        self.workerThread.latestPreviewPixmap.connect(self.updateProcessTab)
        self.workerThread.finished.connect(self.guiChangesOnRenderCompletion)
        self.workerThread.finished.connect(self.workerThread.deleteLater)
        self.workerThread.finished.connect(self.workerThread.quit)
        self.workerThread.finished.connect(
            self.workerThread.wait
        )  # need quit and wait to allow process to exit safely
        self.workerThread.start()

    def splitListIntoStringWithNewLines(self, string_list: list[str]):
        # Join the strings with newline characters
        return "\n".join(string_list)
        # Set the text to the QTextEdit

    def questionToOverride(self):
        reply = QMessageBox.question(
            self.parent,
            "",
            "File exists, do you want to overwrite?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        return reply == QMessageBox.Yes # type: ignore
            

    def run(
        self,
        renderQueue: RenderQueue,
    ):
        self.settings.readSettings()
        self.pausedSharedMemory = shared_memory.SharedMemory(
            name=PAUSED_STATE_SHARED_MEMORY_ID, create=True, size=1
        )
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(False)
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.resumeRender)
        

        if os.path.isfile(renderQueue[0].outputPath):
            self.isOverwrite = self.questionToOverride()
            if not self.isOverwrite:
                self.onRenderCompletion()
                self.guiChangesOnRenderCompletion()
                return  # has to be put at end of function,  so allow for the exit processes to occur"""
        renderOptions = renderQueue[0]
        
        

        # if upscale or interpolate
        """
        Function to start the rendering process
        It will initially check for any issues with the current setup, (invalid file, no permissions, etc..)
        Then, based on the settings selected, it will build a command that is then passed into rve-backend
        Finally, It will handle the render via ffmpeg. Taking in the frames from pipe and handing them into ffmpeg on a sperate thread
        """
        # get model attributes

        if renderOptions.interpolateModel:
            self.interpolateModelFile, interpolateDownloadFile = (
                interpolateModels[renderOptions.interpolateModel][0],
                interpolateModels[renderOptions.interpolateModel][1],
            )
            DownloadModel(
                modelFile=self.interpolateModelFile,
                downloadModelFile=interpolateDownloadFile,
            )
        else:
            renderOptions.interpolateTimes = 1
        if renderOptions.upscaleModel:
            self.upscaleModelFile, upscaleDownloadFile = (
                upscaleModels[renderOptions.upscaleModel][0],
                upscaleModels[renderOptions.upscaleModel][1],
            )
            self.upscaleTimes = upscaleModels[renderOptions.upscaleModel][2]
            self.upscaleModelArch = upscaleModels[renderOptions.upscaleModel][3]
            if self.upscaleModelArch != "custom":  # custom models are not downloaded
                DownloadModel(
                    modelFile=self.upscaleModelFile,
                    downloadModelFile=upscaleDownloadFile,
                )
        else:
            self.upscaleTimes = 1
            self.upscaleModelArch = "custom"
    
        writeThread = Thread(target=lambda: self.renderToPipeThread(renderQueue))
        writeThread.start()
        self.startGUIUpdate()

    def build_command(self, renderOptions: RenderOptions):

        

        if (
                renderOptions.backend == "pytorch (cuda)"
                or renderOptions.backend == "pytorch (rocm)"
            ):
                renderOptions.backend = (
                    "pytorch"  # pytorch is the same for both cuda and rocm
                )

        command = [
                f"{PYTHON_PATH}",
                "-W",
                "ignore",
                os.path.join(BACKEND_PATH, "rve-backend.py"),
                "-i",
                renderOptions.inputFile,
                "-o",
                f"{renderOptions.outputPath}",
                "-b",
                f"{renderOptions.backend}",
                "--precision",
                f"{self.settings.settings['precision']}",
                "--video_encoder_preset",
                f"{self.settings.settings['encoder'].replace(' (experimental)', '').replace(' (40 series and up)','')}",  # remove experimental from encoder
                "--video_pixel_format",
                f"{self.settings.settings['video_pixel_format']}",
                "--audio_encoder_preset",
                f"{self.settings.settings['audio_encoder']}",
                "--audio_bitrate",
                f"{self.settings.settings['audio_bitrate']}",
                "--crf",
                f"{self.qualityToCRF[self.settings.settings['video_quality']]}",
                "--tensorrt_opt_profile",
                f"{self.settings.settings['tensorrt_optimization_level']}",
                "--pause_shared_memory_id",
                f"{PAUSED_STATE_SHARED_MEMORY_ID}",
                "--ncnn_gpu_id",
                f"{self.settings.settings['ncnn_gpu_id']}",
                "--pytorch_gpu_id",
                f"{self.settings.settings['pytorch_gpu_id']}",
            ]

        if renderOptions.upscaleModel:
            modelPath = os.path.join(MODELS_PATH, upscaleModelFile)
            if self.upscaleModelArch == "custom":
                modelPath = os.path.join(CUSTOM_MODELS_PATH, upscaleModelFile)
            command += [
                "--upscale_model",
                modelPath,
            ]
            if renderOptions.tilingEnabled:
                command += [
                    "--tilesize",
                    f"{renderOptions.tilesize}",
                ]

        if renderOptions.interpolateModel:
            command += [
                "--interpolate_model",
                os.path.join(
                    MODELS_PATH,
                    interpolateModelFile,
                ),
                "--interpolate_factor",
                f"{renderOptions.interpolateTimes}",
            ]
            if renderOptions.sloMoMode:
                command += [
                    "--slomo_mode",
                ]
            if renderOptions.dyanmicScaleOpticalFlow:
                command += [
                    "--dynamic_scaled_optical_flow",
                ]
            if renderOptions.ensemble:
                command += [
                    "--ensemble",
                ]
        if self.settings.settings["auto_border_cropping"] == "True":
            command += [
                "--border_detect",
            ]

        if self.settings.settings["preview_enabled"] == "True":
            command += [
                "--preview_shared_memory_id",
                f"{IMAGE_SHARED_MEMORY_ID}",
            ]

        if self.settings.settings["scene_change_detection_enabled"] == "False":
            command += ["--scene_detect_method", "none"]
        else:
            command += [
                "--scene_detect_method",
                self.settings.settings["scene_change_detection_method"],
                "--scene_detect_threshold",
                self.settings.settings["scene_change_detection_threshold"],
            ]

        if renderOptions.benchmarkMode:
            command += ["--benchmark"]

        if self.settings.settings["uhd_mode"] == "True":
            if renderOptions.videoWidth > 1920 or renderOptions.videoHeight > 1080:
                command += ["--UHD_mode"]
                log("UHD mode enabled")

        if self.isOverwrite:
            command += ["--overwrite"]

        return command
    
    def renderToPipeThread(
        self,
        renderQueue: list[RenderOptions],
    ):
        for renderOptions in renderQueue:
            

            # get video attributes
            self.outputVideoWidth = renderOptions.videoWidth * self.upscaleTimes
            self.outputVideoHeight = renderOptions.videoHeight * self.upscaleTimes

            # discord rpc
            if self.settings.settings["discord_rich_presence"] == "True":
                try:
                    self.discordRPC = DiscordRPC()
                    self.discordRPC.start_discordRPC(
                        "Enhancing",
                        os.path.basename(renderOptions.inputFile),
                        renderOptions.backend,
                    )
                except Exception:
                    pass
            # builds command
            command = self.build_command(renderOptions)            

            self.renderProcess = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            textOutput = []
            for line in iter(self.renderProcess.stdout.readline, b""):
                if self.renderProcess.poll() is not None:
                    break  # Exit the loop if the process has terminated

                line = str(line.strip())
                if "it/s" in line:
                    textOutput = textOutput[:-1]
                if "FPS" in line:
                    textOutput = textOutput[
                        :-1
                    ]  # slice the list to only get the last updated data
                    self.currentFrame = int(
                        re.search(r"Current Frame: (\d+)", line).group(1)
                    )
                if any(char.isalpha() for char in line):
                    textOutput.append(line)
                # self.setRenderOutputContent(textOutput)
                self.renderTextOutputList = textOutput.copy()
                if "Time to complete render" in line:
                    break
            for line in textOutput:
                if len(line) > 2:
                    log(line)
        self.onRenderCompletion()

    def guiChangesOnRenderCompletion(self):
        # Have to swap the visibility of these here otherwise crash for some reason
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setEnabled(True)
        self.parent.previewLabel.clear()
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        self.parent.enableProcessPage()
        self.parent.startRenderButton.setVisible(True)

    def onRenderCompletion(self):
        try:
            self.renderProcess.wait()
        except Exception:
            pass
        # Have to swap the visibility of these here otherwise crash for some reason
        if (
            self.settings.settings["discord_rich_presence"] == "True"
        ):  # only close if it exists
            self.discordRPC.closeRPC()
        try:
            self.workerThread.stop()
            self.workerThread.quit()
            self.workerThread.wait()
        except Exception:
            pass  # pass just incase internet error caused a skip

    def getRoundedPixmap(self, pixmap, corner_radius):
        size = pixmap.size()
        mask = QPixmap(size)
        mask.fill(Qt.transparent)  # type: ignore

        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore

        path = QPainterPath()
        path.addRoundedRect(
            0, 0, size.width(), size.height(), corner_radius, corner_radius
        )

        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        rounded_pixmap = QPixmap(size)
        rounded_pixmap.fill(Qt.transparent)  # type: ignore

        painter = QPainter(rounded_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore
        painter.setRenderHint(QPainter.SmoothPixmapTransform)  # type: ignore
        painter.drawPixmap(0, 0, mask)
        painter.end()

        return rounded_pixmap

    def modelNameToFile(self):
        pass

    def updateProcessTab(self, qimage: QtGui.QImage):
        """
        Called by the worker QThread, and updates the GUI elements: Progressbar, Preview, FPS
        """

        if self.renderTextOutputList is not None:
            # print(self.renderTextOutputList)
            self.parent.renderOutput.setPlainText(
                self.splitListIntoStringWithNewLines(self.renderTextOutputList)
            )
            scrollbar = self.parent.renderOutput.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            self.parent.progressBar.setValue(self.currentFrame)
        if not qimage.isNull():
            label_width = self.parent.previewLabel.width()
            label_height = self.parent.previewLabel.height()

            p = qimage.scaled(
                label_width, label_height, Qt.AspectRatioMode.KeepAspectRatio
            )  # type: ignore
            pixmap = QtGui.QPixmap.fromImage(p)

            roundedPixmap = self.getRoundedPixmap(pixmap, corner_radius=10)
            self.parent.previewLabel.setPixmap(roundedPixmap)
