import subprocess
import os
from threading import Thread
import re

from PySide6 import QtGui
from PySide6.QtGui import QPixmap, QPainter, QPainterPath, QColor
from PySide6.QtCore import Qt, QSize
from ..BuildFFmpegCommand import BuildFFMpegCommand

from .AnimationHandler import AnimationHandler
from .QTcustom import UpdateGUIThread, RegularQTPopup, show_layout_widgets, hide_layout_widgets
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
)


class ProcessTab:
    def __init__(self, parent, gmfssSupport: bool):
        self.parent = parent
        self.imagePreviewSharedMemoryID = "/image_preview" + str(os.getpid())
        self.renderTextOutputList = None
        self.currentFrame = 0
        self.animationHandler = AnimationHandler()
        self.tileUpAnimationHandler = AnimationHandler()
        self.tileDownAnimationHandler = AnimationHandler()
        self.gmfssSupport = gmfssSupport
        # encoder dict
        # key is the name in RVE gui
        # value is the encoder used

        # get default backend
        self.QConnect()
        self.switchInterpolationAndUpscale()

    def getTotalModels(self, method: str, backend: str) -> dict:
        """
        returns
        the current models available given a method (interpolate, upscale) and a backend (ncnn, tensorrt, pytorch)
        """
        log("Getting total models, method: " + method + " backend: " + backend)
        if method == "Interpolate":
            match backend:
                case "ncnn":
                    models = ncnnInterpolateModels
                case "pytorch":
                    models = pytorchInterpolateModels
                case "tensorrt":
                    models = tensorrtInterpolateModels
                case "directml":
                    models = onnxInterpolateModels
                case _:
                    RegularQTPopup(
                        "Failed to import any backends!, please try to reinstall the app!"
                    )
                    errorAndLog("Failed to import any backends!")
                    models = None
            self.parent.interpolationContainer.setVisible(True)
        if method == "Upscale":
            match backend:
                case "ncnn":
                    models = ncnnUpscaleModels
                case "pytorch":
                    models = pytorchUpscaleModels
                case "tensorrt":
                    models = tensorrtUpscaleModels
                case "directml":
                    models = onnxUpscaleModels
                case _:
                    RegularQTPopup(
                        "Failed to import any backends!, please try to reinstall the app!"
                    )
                    errorAndLog("Failed to import any backends!")
                    models = None
        if method == "Denoise":
            match backend:
                case "ncnn":
                    models = None
                case "pytorch":
                    models = pytorchDenoiseModels
                case "tensorrt":
                    models = None
                case "directml":
                    models = None
                case _:
                    RegularQTPopup(
                        "Failed to import any backends!, please try to reinstall the app!"
                    )
                    errorAndLog("Failed to import any backends!")
                    models = None
        return models

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
        cbs = (self.parent.methodComboBox, self.parent.backendComboBox)
        for combobox in cbs:
            combobox.currentIndexChanged.connect(self.switchInterpolationAndUpscale)
        # set tile size visible to false by default
        self.parent.tileSizeContainer.setVisible(False)
        # connect up tilesize container visiable
        self.parent.tilingCheckBox.stateChanged.connect(self.onTilingSwitch)

        self.parent.interpolationMultiplierSpinBox.valueChanged.connect(
            self.parent.updateVideoGUIDetails
        )
        self.parent.modelComboBox.currentIndexChanged.connect(
            self.parent.updateVideoGUIDetails
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

    def switchInterpolationAndUpscale(self):
        """
        Called every render, gets the correct model based on the backend and the method.
        """

        self.parent.modelComboBox.clear()
        # overwrite method
        method = self.parent.methodComboBox.currentText()
        backend = self.parent.backendComboBox.currentText()
        models = self.getTotalModels(method=method, backend=backend)
        if backend != "pytorch":
            self.parent.methodComboBox.removeItem(self.parent.methodComboBox.findText("Denoise"))
        elif self.parent.methodComboBox.findText("Denoise") == -1 and backend == 'pytorch':
            self.parent.methodComboBox.addItem("Denoise")
        self.parent.modelComboBox.addItems(models)
        total_items = self.parent.modelComboBox.count()
        if total_items > 0 and method.lower() == "interpolate":
            self.parent.modelComboBox.setCurrentIndex(total_items - 1)

        if method.lower() == "interpolate":
            self.parent.interpolationContainer.setVisible(True)
            self.parent.upscaleContainer.setVisible(False)
            self.animationHandler.dropDownAnimation(self.parent.interpolationContainer)
            if not self.gmfssSupport:
                # Disable specific options based on the selected text
                for i in range(self.parent.modelComboBox.count()):
                    if (
                        self.parent.modelComboBox.itemText(i)
                        == "GMFSS (Slowest Model, Animation)"
                    ):  # hacky solution, just straight copy pasted
                        self.parent.modelComboBox.model().item(i).setEnabled(
                            self.gmfssSupport
                        )
        elif method.lower() == "upscale" or method.lower() == "denoise":
            self.parent.interpolationContainer.setVisible(False)
            self.parent.upscaleContainer.setVisible(True)
            
            self.animationHandler.dropDownAnimation(self.parent.upscaleContainer)

        self.parent.updateVideoGUIDetails()

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
        method: str,
        backend: str,
        interpolationTimes: int,
        model: str,
        benchmarkMode: bool,
    ):
        self.inputFile = inputFile
        self.outputPath = outputPath
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight
        self.videoFps = videoFps
        self.tilingEnabled = tilingEnabled
        self.tilesize = tilesize
        self.videoFrameCount = videoFrameCount
        models = self.getTotalModels(method=method, backend=backend)

        # if upscale or interpolate
        """
        Function to start the rendering process
        It will initially check for any issues with the current setup, (invalid file, no permissions, etc..)
        Then, based on the settings selected, it will build a command that is then passed into rve-backend
        Finally, It will handle the render via ffmpeg. Taking in the frames from pipe and handing them into ffmpeg on a sperate thread
        """
        self.benchmarkMode = benchmarkMode
        # get model attributes
        self.modelFile = models[model][0]
        self.downloadFile = models[model][1]
        self.upscaleTimes = models[model][2]
        self.modelArch = models[model][3]

        # get video attributes
        self.outputVideoWidth = videoWidth * self.upscaleTimes
        self.outputVideoHeight = videoHeight * self.upscaleTimes
        
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
                    method, os.path.basename(self.inputFile), backend
                )
            except Exception:
                pass
        if self.modelArch != "custom":  # custom models are not downloaded
            DownloadModel(
                modelFile=self.modelFile,
                downloadModelFile=self.downloadFile,
                backend=backend,
            )
        # self.ffmpegWriteThread()

        writeThread = Thread(
            target=lambda: self.renderToPipeThread(
                method=method, backend=backend, interpolateTimes=interpolationTimes
            )
        )
        writeThread.start()
        self.startGUIUpdate()

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

    def renderToPipeThread(self, method: str, backend: str, interpolateTimes: int):
        # builds command

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
            "--pausedFile",
            f"{self.pausedFile}",
        ]
        if method == "Upscale" or method == "Denoise":
            modelPath = os.path.join(MODELS_PATH, self.modelFile)
            if self.modelArch == "custom":
                modelPath = os.path.join(CUSTOM_MODELS_PATH, self.modelFile)
            command += [
                "--upscaleModel",
                modelPath,
                "--interpolateFactor",
                "1",
            ]
            if self.tilingEnabled:
                command += [
                    "--tilesize",
                    f"{self.tilesize}",
                ]
        if method == "Interpolate":
            command += [
                "--interpolateModel",
                os.path.join(
                    MODELS_PATH,
                    self.modelFile,
                ),
                "--interpolateFactor",
                f"{interpolateTimes}",
            ]
        if self.settings["preview_enabled"] == "True":
            command += [
                "--shared_memory_id",
                f"{self.imagePreviewSharedMemoryID}",
            ]
        if self.settings["scene_change_detection_enabled"] == "False":
            command += ["--sceneDetectMethod", "none"]
        else:
            command += [
                "--sceneDetectMethod",
                self.settings["scene_change_detection_method"],
                "--sceneDetectSensitivity",
                self.settings["scene_change_detection_threshold"],
            ]
        if self.benchmarkMode:
            command += ["--benchmark"]
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
            if "torch_tensorrt.dynamo" in line:
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
        log(str(textOutput))
        self.onRenderCompletion()

    def onRenderCompletion(self):
        self.renderProcess.wait()
        # Have to swap the visibility of these here otherwise crash for some reason
        hide_layout_widgets(self.parent.onRenderButtonsContiainer)
        self.parent.startRenderButton.setVisible(True)
        self.parent.startRenderButton.setEnabled(True)
        if self.settings["discord_rich_presence"] == "True":  # only close if it exists
            self.discordRPC.closeRPC()
        try:
            self.workerThread.stop()
            self.workerThread.quit()
            self.workerThread.wait()
        except Exception:
            pass  # pass just incase internet error caused a skip
        # reset image preview
        self.parent.previewLabel.clear()
        self.parent.startRenderButton.clicked.disconnect()

        self.parent.startRenderButton.clicked.connect(self.parent.startRender)

        self.parent.enableProcessPage()

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
