import subprocess
import os
from threading import Thread
import re

from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QColor
from PySide6.QtCore import Qt, QSize
from ..BuildFFmpegCommand import BuildFFMpegCommand

from .AnimationHandler import AnimationHandler
from .QTcustom import (
    UpdateGUIThread,
    RegularQTPopup,
    show_layout_widgets,
    hide_layout_widgets,
)
from ..constants import BACKEND_PATH, PYTHON_PATH, MODELS_PATH, CUSTOM_MODELS_PATH
from ..Util import (
    currentDirectory,
    log,
    log,
    errorAndLog,
)
from ..DownloadModels import DownloadModel
from .SettingsTab import Settings
from ..DiscordRPC import DiscordRPC
from ..ModelHandler import (
    ncnnInterpolateModels,
    ncnnUpscaleModels,
    pytorchInterpolateModels,
    pytorchUpscaleModels,
    pytorchDenoiseModels,
    tensorrtInterpolateModels,
    tensorrtUpscaleModels,
    onnxUpscaleModels,
    onnxInterpolateModels,
    totalModels,
)


class ProcessTab:
    def __init__(self, parent):
        self.parent = parent
        self.imagePreviewSharedMemoryID = "/image_preview" + str(os.getpid())
        self.renderTextOutputList = None
        self.currentFrame = 0
        self.animationHandler = AnimationHandler()
        self.tileUpAnimationHandler = AnimationHandler()
        self.tileDownAnimationHandler = AnimationHandler()
        # encoder dict
        # key is the name in RVE gui
        # value is the encoder used

        # get default backend
        self.QConnect()
        self.populateModels(self.parent.backendComboBox.currentText())

    def getModels(self, backend):
        """
        returns models based on backend, used for populating the model comboboxes [interpolate, upscale]
        """
        match backend:
            case "ncnn":
                interpolateModels = ncnnInterpolateModels
                upscaleModels = ncnnUpscaleModels
            case "pytorch (cuda)":
                interpolateModels = pytorchInterpolateModels
                upscaleModels = pytorchUpscaleModels
            case "pytorch (rocm)":
                interpolateModels = pytorchInterpolateModels
                upscaleModels = pytorchUpscaleModels
            case "tensorrt":
                interpolateModels = tensorrtInterpolateModels
                upscaleModels = tensorrtUpscaleModels
            case "directml":
                interpolateModels = onnxInterpolateModels
                upscaleModels = onnxUpscaleModels
            case _:
                RegularQTPopup(
                    "Failed to import any backends!, please try to reinstall the app!"
                )
                errorAndLog("Failed to import any backends!")
                return {}
        return interpolateModels, upscaleModels

    def populateModels(self, backend) -> dict:
        """
        returns
        the current models available given a method (interpolate, upscale) and a backend (ncnn, tensorrt, pytorch)
        """
        interpolateModels, upscaleModels = self.getModels(backend)
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

        self.parent.inputFileSelectButton.clicked.connect(self.parent.openInputFile)
        self.parent.inputFileText.textChanged.connect(self.parent.loadVideo)
        self.parent.outputFileSelectButton.clicked.connect(self.parent.openOutputFolder)
        # connect render button
        self.parent.startRenderButton.clicked.connect(self.parent.startRender)
        # set tile size visible to false by default
        self.parent.tileSizeContainer.setVisible(False)
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
        with open(self.pausedFile, "w") as f:
            f.write("True")
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(True)
        self.parent.startRenderButton.setEnabled(True)

    def resumeRender(self):
        with open(self.pausedFile, "w") as f:
            f.write("False")
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.onRenderButtonsContiainer.setEnabled(True)
        self.parent.startRenderButton.setVisible(False)

    def startGUIUpdate(self):
        self.workerThread = UpdateGUIThread(
            parent=self,
            imagePreviewSharedMemoryID=self.imagePreviewSharedMemoryID,
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

    def run(
        self,
        inputFile: str,
        outputPath: str,
        videoWidth: int,
        videoHeight: int,
        videoFps: float,
        videoFrameCount: int,
        tilesize: int,
        tilingEnabled: bool,
        backend: str,
        interpolateTimes: int,
        upscaleModel: str,
        interpolateModel: str,
        benchmarkMode: bool,
    ):
        interpolateModels, upscaleModels = self.getModels(backend)
        if interpolateModel == "None":
            interpolateModel = None
            interpolateModelFile = None
        if upscaleModel == "None":
            upscaleModel = None
            upscaleModelFile = None
        self.inputFile = inputFile
        self.outputPath = outputPath
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.videoFps = videoFps
        self.tilingEnabled = tilingEnabled
        self.tilesize = tilesize
        self.videoFrameCount = videoFrameCount

        # if upscale or interpolate
        """
        Function to start the rendering process
        It will initially check for any issues with the current setup, (invalid file, no permissions, etc..)
        Then, based on the settings selected, it will build a command that is then passed into rve-backend
        Finally, It will handle the render via ffmpeg. Taking in the frames from pipe and handing them into ffmpeg on a sperate thread
        """
        self.benchmarkMode = benchmarkMode
        # get model attributes

        if interpolateModel:
            interpolateModelFile, interpolateDownloadFile = (
                interpolateModels[interpolateModel][0],
                interpolateModels[interpolateModel][1],
            )
        else:
            interpolateTimes = 1
        if upscaleModel:
            upscaleModelFile, upscaleDownloadFile = (
                upscaleModels[upscaleModel][0],
                upscaleModels[upscaleModel][1],
            )
            upscaleTimes = upscaleModels[upscaleModel][2]
            upscaleModelArch = upscaleModels[upscaleModel][3]
        else:
            upscaleTimes = 1
            upscaleModelArch = "custom"

        if interpolateModel:
            DownloadModel(
                modelFile=interpolateModelFile,
                downloadModelFile=interpolateDownloadFile,
            )
        if upscaleModelArch != "custom":  # custom models are not downloaded
            if upscaleModelFile:
                DownloadModel(
                    modelFile=upscaleModelFile, downloadModelFile=upscaleDownloadFile
                )
        # get video attributes
        self.outputVideoWidth = videoWidth * upscaleTimes
        self.outputVideoHeight = videoHeight * upscaleTimes

        # set up pausing
        self.pausedFile = os.path.join(
            currentDirectory(), os.path.basename(inputFile) + "_pausedState.txt"
        )
        show_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(False)
        self.parent.startRenderButton.clicked.disconnect()
        self.parent.startRenderButton.clicked.connect(self.resumeRender)

        # get most recent settings
        settings = Settings()
        settings.readSettings()
        self.settings = settings.settings

        # get built ffmpeg command
        buildFFMpegCommand = BuildFFMpegCommand(
            encoder=self.settings["encoder"], quality=self.settings["video_quality"]
        )
        self.buildFFMpegsettings = buildFFMpegCommand.buildFFmpeg()

        # discord rpc
        if self.settings["discord_rich_presence"] == "True":
            try:
                self.discordRPC = DiscordRPC()
                self.discordRPC.start_discordRPC(
                    "Enhancing", os.path.basename(self.inputFile), backend
                )
            except Exception:
                pass

        writeThread = Thread(
            target=lambda: self.renderToPipeThread(
                backend=backend,
                interpolateTimes=interpolateTimes,
                interpolateModelFile=interpolateModelFile,
                upscaleModelFile=upscaleModelFile,
                upscaleModelArch=upscaleModelArch,
            )
        )
        writeThread.start()
        self.startGUIUpdate()

    def renderToPipeThread(
        self,
        backend: str,
        interpolateTimes: int,
        interpolateModelFile: str,
        upscaleModelFile: str,
        upscaleModelArch: str,
    ):
        # builds command
        if backend == "pytorch (cuda)" or backend == "pytorch (rocm)":
            backend = "pytorch"  # pytorch is the same for both cuda and rocm

        command = [
            f"{PYTHON_PATH}",
            "-W",
            "ignore",
            os.path.join(BACKEND_PATH, "rve-backend.py"),
            "-i",
            self.inputFile,
            "-o",
            f"{self.outputPath}",
            "-b",
            f"{backend}",
            "--precision",
            f"{self.settings['precision']}",
            "--custom_encoder",
            f"{self.buildFFMpegsettings}",
            "--tensorrt_opt_profile",
            f"{self.settings['tensorrt_optimization_level']}",
            "--paused_file",
            f"{self.pausedFile}",
        ]

        if upscaleModelFile:
            modelPath = os.path.join(MODELS_PATH, upscaleModelFile)
            if upscaleModelArch == "custom":
                modelPath = os.path.join(CUSTOM_MODELS_PATH, upscaleModelFile)
            command += [
                "--upscale_model",
                modelPath,
            ]
            if self.tilingEnabled:
                command += [
                    "--tilesize",
                    f"{self.tilesize}",
                ]

        if interpolateModelFile:
            command += [
                "--interpolate_model",
                os.path.join(
                    MODELS_PATH,
                    interpolateModelFile,
                ),
                "--interpolate_factor",
                f"{interpolateTimes}",
            ]

        if self.settings["preview_enabled"] == "True":
            command += [
                "--shared_memory_id",
                f"{self.imagePreviewSharedMemoryID}",
            ]

        if self.settings["scene_change_detection_enabled"] == "False":
            command += ["--scene_detect_method", "none"]
        else:
            command += [
                "--scene_detect_method",
                self.settings["scene_change_detection_method"],
                "--scene_detect_threshold",
                self.settings["scene_change_detection_threshold"],
            ]

        if self.benchmarkMode:
            command += ["--benchmark"]
        
        if self.settings["uhd_mode"] == "True":
            if self.videoWidth > 1920 or self.videoHeight > 1080:
                command += ["--UHD_mode"]
                log("UHD mode enabled")


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

            # filter out lines from logs here
            if "torch_tensorrt.dynamo" in line:
                continue
            if "INFO:torch_tensorrt" in line:
                continue
            if "WARNING: [Torch-TensorRT]" in line:
                continue
            if "Unable to import quantization" in line:
                continue

            line = str(line.strip())
            if "it/s" in line:
                textOutput = textOutput[:-1]
            if "FPS" in line:
                textOutput = textOutput[
                    :-2
                ]  # slice the list to only get the last updated data
                self.currentFrame = int(
                    re.search(r"Current Frame: (\d+)", line).group(1)
                )
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
        self.parent.processSettingsContainer.setEnabled(True)
        self.parent.startRenderButton.setVisible(True)

    def onRenderCompletion(self):
        self.renderProcess.wait()
        # Have to swap the visibility of these here otherwise crash for some reason
        if self.settings["discord_rich_presence"] == "True":  # only close if it exists
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
