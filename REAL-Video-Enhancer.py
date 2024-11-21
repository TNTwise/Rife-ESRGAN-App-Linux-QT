import sys
import os
from threading import Thread
# patch for macos
if sys.platform == "darwin":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # this goes one step up, and goes into the actual directory. This is where backend will be copied to.
    os.chdir("..")
import math
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtGui import QIcon
from src.Util import log
from mainwindow import Ui_MainWindow
from PySide6 import QtSvg  # Import the QtSvg module so svg icons can be used on windows
from src.version import version
from src.InputHandler import VideoLoader
from src.ModelHandler import getCustomModelScale

# other imports
from src.Util import (
    getOSInfo,
    get_gpu_info,
    getRAMAmount,
    getCPUInfo,
    checkForWritePermissions,
    getAvailableDiskSpace,
    copyFile,
    createDirectory,
)
from src.constants import CUSTOM_MODELS_PATH
from src.ui.ProcessTab import ProcessTab
from src.ui.DownloadTab import DownloadTab
from src.ui.SettingsTab import SettingsTab, Settings
from src.ui.HomeTab import HomeTab
from src.Backendhandler import BackendHandler
from src.ModelHandler import totalModels
from src.ui.AnimationHandler import AnimationHandler
from src.ui.QTstyle import Palette
from src.ui.QTcustom import RegularQTPopup


class MainWindow(QMainWindow, Ui_MainWindow):
    """Main window class for the REAL Video Enhancer application.

    This class extends the QMainWindow and Ui_MainWindow classes to create the main window of the application.
    It sets up the user interface, connects buttons to switch menus, and handles various functionalities such as rendering, file selection, and backend setup.

    Attributes:
        homeDir (str): The home directory path.
        interpolateTimes (int): The number of times to interpolate frames.
        upscaleTimes (int): The number of times to upscale frames.
        pipeInFrames (None): Placeholder for input frames.
        latestPreviewImage (None): Placeholder for the latest preview image.
        aspect_ratio (float): The aspect ratio of the window.

    Methods:
        __init__(): Initializes the MainWindow class.
        QButtonConnect(): Connects buttons to switch menus.
        setupBackendDeps(): Sets up the backend dependencies.
        switchToHomePage(): Switches to the home page.
        switchToProcessingPage(): Switches to the processing page.
        switchToSettingsPage(): Switches to the settings page.
        switchToDownloadPage(): Switches to the download page.
        recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(): Recursively checks if at least one backend is installed.
        startRender(): Starts the rendering process.
        disableProcessPage(): Disables the process page.
        enableProcessPage(): Enables the process page.
        getAvailableBackends(): Retrieves the available backends.
        openInputFile(): Opens an input video file.
        openOutputFolder(): Opens an output folder.
        killRenderProcess(): Terminates the render process.
        closeEvent(event): Handles the close event of the main window."""

    def __init__(self):
        super().__init__()

        # set up base variables
        self.homeDir = os.path.expanduser("~")
        self.pipeInFrames = None
        self.latestPreviewImage = None
        self.videoWidth = None
        self.videoHeight = None
        self.isVideoLoaded = False

        # setup application

        # Set up the user interface from Designer.
        self.setupUi(self)
        backendHandler = BackendHandler(self)
        backendHandler.enableCorrectBackends()

        backendHandler.setupBackendDeps()
        self.backends, self.fullOutput = (
            backendHandler.recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(
                firstIter=True
            )
        )

        backendHandler.hideUninstallButtons()
        backendHandler.showUninstallButton(self.backends)
        icon_path = ":/icons/icons/logo-v2.svg"
        self.setWindowIcon(QIcon(icon_path))
        QApplication.setWindowIcon(QIcon(icon_path))
        self.setWindowTitle("REAL Video Enhancer")
        self.setPalette(QApplication.style().standardPalette())
        self.setMinimumSize(1100, 700)

        self.aspect_ratio = self.width() / self.height()

        # set default home page
        self.stackedWidget.setCurrentIndex(0)

        self.QConnect()
        # set up tabs
        self.backendComboBox.addItems(self.backends)
        printOut = (
            "------REAL Video Enhancer------\n"
            + "System Information: \n"
            + "OS: "
            + getOSInfo()
            + "\n"
            + "CPU: "
            + getCPUInfo()
            + "\n"
            + "GPU: "
            + get_gpu_info()
            + "\n"
            + "RAM: "
            + getRAMAmount()
            + "\n"
            + "Available Disk Space: "
            + str(round(getAvailableDiskSpace(), 2))
            + "GB"
            + "\n"
            + "-------------------------------------------\n"
            + "Software Information: \n"
            + f"REAL Video Enhancer Version: {version}\n"
            + self.fullOutput
        )
        self.systemInfoText.setText(printOut)
        log(printOut)

        # process the output
        for line in self.fullOutput.lower().split("\n"):
            if "half precision support:" in line:
                halfPrecisionSupport = "true" in line
            if "gmfss support:" in line:
                gmfssSupport = "true" in line
        settings = Settings()
        settings.readSettings()
        self.settings = settings
        self.processTab = ProcessTab(
            parent=self,
            gmfssSupport=gmfssSupport,
        )
        self.homeTab = HomeTab(parent=self)
        self.downloadTab = DownloadTab(parent=self)
        self.settingsTab = SettingsTab(
            parent=self, halfPrecisionSupport=halfPrecisionSupport
        )

        # Startup Animation
        self.animationHandler = AnimationHandler()
        self.animationHandler.fadeInAnimation(self)

    def QConnect(self):
        # connect buttons to switch menus
        self.homeBtn.clicked.connect(self.switchToHomePage)
        self.processBtn.clicked.connect(self.switchToProcessingPage)
        self.settingsBtn.clicked.connect(self.switchToSettingsPage)
        self.downloadBtn.clicked.connect(self.switchToDownloadPage)
        # connect getting default output file
        

    def setButtonsUnchecked(self, buttonToIgnore):
        buttons = [
            self.homeBtn,
            self.processBtn,
            self.settingsBtn,
            self.downloadBtn,
        ]
        for button in buttons:
            if button != buttonToIgnore:
                button.setChecked(False)
            else:
                button.setChecked(True)

    # switch menus
    def switchToHomePage(self):
        self.animationHandler.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.homePage)
        self.setButtonsUnchecked(self.homeBtn)
        self.animationHandler.fadeInAnimation(self.stackedWidget)

    def switchToProcessingPage(self):
        self.animationHandler.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.procPage)
        self.setButtonsUnchecked(self.processBtn)
        self.animationHandler.fadeInAnimation(self.stackedWidget)

    def switchToSettingsPage(self):
        self.animationHandler.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.settingsPage)
        self.setButtonsUnchecked(self.settingsBtn)
        self.animationHandler.fadeInAnimation(self.stackedWidget)

    def switchToDownloadPage(self):
        self.animationHandler.fadeOutAnimation(self.stackedWidget)
        self.stackedWidget.setCurrentWidget(self.downloadPage)
        self.setButtonsUnchecked(self.downloadBtn)
        self.animationHandler.fadeInAnimation(self.stackedWidget)


    def updateVideoGUIText(self):
        if self.isVideoLoaded:
            modelName = self.modelComboBox.currentText()
            method = self.methodComboBox.currentText()
            interpolateTimes = self.getInterpolateTimes(method, modelName)
            scale = self.getScale(method, modelName)
            text = (
                f"FPS: {round(self.videoFps,0)} -> {round(self.videoFps*interpolateTimes,0)}\n"
                + f"Resolution: {self.videoWidth}x{self.videoHeight} -> {self.videoWidth*scale}x{self.videoHeight*scale}\n"
                + f"Frame Count: {self.videoFrameCount} -> {int(round(self.videoFrameCount * interpolateTimes,0))}\n"
                + f"Bitrate: {self.videoBitrate}\n"
                + f"Encoder: {self.videoEncoder}\n"
                + f"Container: {self.videoContainer}\n"
            )
            self.videoInfoTextEdit.setFontPointSize(10)
            self.videoInfoTextEdit.setText(text)

    def setDefaultOutputFile(self, outputDirectory):
        """
        Sets the default output file for the video enhancer.
        Parameters:
        - useDefaultVideoPath (bool): Flag indicating whether to use the default video path for the output file.
        Returns:
        None
        """
        
        # check if there is a video loaded
        if self.isVideoLoaded:
            inputFile = self.inputFileText.text()
            modelName = self.modelComboBox.currentText()
            method = self.methodComboBox.currentText()
            interpolateTimes = self.getInterpolateTimes(method, modelName)
            scale = self.getScale(method, modelName)

            
            file_name = os.path.splitext(os.path.basename(inputFile))[0]
            output_file = os.path.join(
                outputDirectory,
                f"{file_name}_{interpolateTimes*self.videoFps}fps_{scale*self.videoWidth}x{scale*self.videoHeight}.mkv",
            )
            iteration = 0
            while os.path.isfile(output_file):
                output_file = os.path.join(
                    outputDirectory,
                    f"{file_name}_{interpolateTimes*self.videoFps}fps_{scale*self.videoWidth}x{scale*self.videoHeight}_({iteration}).mkv",
                )
                iteration += 1
            self.outputFileText.setText(output_file)
            return output_file

    def updateVideoGUIDetails(self):
        self.settings.readSettings()
        self.setDefaultOutputFile(self.settings.settings["output_folder_location"])
        self.updateVideoGUIText()

    def getScale(self, method, modelName):
        if method == "Upscale" or method == "Denoise":
            scale = totalModels[modelName][2]
        elif method == "Interpolate":
            scale = 1
        return scale

    def getInterpolateTimes(self, method, modelName):
        if method == "Upscale" or method == "Denoise":
            interpolateTimes = 1
        elif method == "Interpolate":
            interpolateTimes = self.interpolationMultiplierSpinBox.value()
        return interpolateTimes

    def startRender(self):
        if self.isVideoLoaded:
            if checkForWritePermissions(os.path.dirname(self.outputFileText.text())):
                self.startRenderButton.setEnabled(False)
                method = self.methodComboBox.currentText()
                self.progressBar.setRange(
                    0,
                    # only set the range to multiply the frame count if the method is interpolate
                    int(
                        self.videoFrameCount
                        * math.ceil(self.interpolationMultiplierSpinBox.value())
                    )
                    if method == "Interpolate"
                    else self.videoFrameCount,
                )
                self.disableProcessPage()

                self.processTab.run(
                    inputFile=self.inputFileText.text(),
                    outputPath=self.outputFileText.text(),
                    videoWidth=self.videoWidth,
                    videoHeight=self.videoHeight,
                    videoFps=self.videoFps,
                    tilingEnabled=self.tilingCheckBox.isChecked(),
                    tilesize=self.tileSizeComboBox.currentText(),
                    videoFrameCount=self.videoFrameCount,
                    method=method,
                    backend=self.backendComboBox.currentText(),
                    interpolationTimes=self.interpolationMultiplierSpinBox.value(),
                    model=self.modelComboBox.currentText(),
                    benchmarkMode=self.benchmarkModeCheckBox.isChecked(),
                )
            else:
                RegularQTPopup("No write permissions to the output directory!")
        else:
            pass
            RegularQTPopup("Please select a video file!")

    def disableProcessPage(self):
        self.processSettingsContainer.setDisabled(True)

    def enableProcessPage(self):
        self.processSettingsContainer.setEnabled(True)

    def loadVideo(self, inputFile):
        
        videoHandler = VideoLoader(inputFile)
        videoHandler.loadVideo()
        if not videoHandler.isValidVideo(): # this handles case for invalid youtube link and invalid video file
            RegularQTPopup("Not a valid input!")
            return
        videoHandler.getData()
        self.videoWidth = videoHandler.width
        self.videoHeight = videoHandler.height
        self.videoFps = videoHandler.fps
        self.videoLength = videoHandler.duration
        self.videoFrameCount = videoHandler.total_frames
        self.videoEncoder = videoHandler.codec_str
        self.videoBitrate = videoHandler.bitrate
        self.videoContainer = videoHandler.videoContainer
        
        self.inputFileText.setText(inputFile)
        self.outputFileText.setEnabled(True)
        self.outputFileSelectButton.setEnabled(True)
        self.isVideoLoaded = True
        self.updateVideoGUIDetails()

    # input file button
    def openInputFile(self):
        """
        Opens a video file and checks if it is valid,

        if it is valid, it will set self.inputFile to the input file, and set the text input field to the input file path.
        if it is not valid, it will give a warning to the user.

        > IMPLEMENT AFTER SELECT AI >  Last, It will enable the output select button, and auto create a default output file

        *NOTE
        This function will set self.videoWidth, self.videoHeight, and self.videoFps

        """

        fileFilter = "Video files (*.mp4 *.mov *.webm *.mkv)"
        inputFile, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select File",
            dir=self.homeDir,
            filter=fileFilter,
        )
        self.loadVideo(inputFile)

    def importCustomModel(self, format: str):
        """
        *args
        format: str
            The format of the model to import (pytorch, ncnn)
        """

        if format == "pytorch":
            fileFilter = "PyTorch Model (*.pth)"

            modelFile, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Select PyTorch Model",
                dir=self.homeDir,
                filter=fileFilter,
            )
            if getCustomModelScale(os.path.basename(modelFile)):
                outputModelPath = os.path.join(
                    CUSTOM_MODELS_PATH, os.path.basename(modelFile)
                )
                copyFile(modelFile, CUSTOM_MODELS_PATH)
                if os.path.isfile(outputModelPath):
                    RegularQTPopup(
                        "Model imported successfully!\nPlease restart the app for the changes to take effect."
                    )
                else:
                    RegularQTPopup("Failed to import model!\nPlease try again.")
            else:
                RegularQTPopup(
                    "Custom model does not have a valid\nupscale factor in the name.\nExample: 2x or x2. Skipping import..."
                )

        elif format == "ncnn":
            binFileFilter = "NCNN Bin (*.bin)"
            modelBinFile, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Select NCNN Bin",
                dir=self.homeDir,
                filter=binFileFilter,
            )
            if getCustomModelScale(os.path.basename(modelFile)):
                if modelBinFile == "":
                    RegularQTPopup("Please select a bin file!")
                    return
                modelParamFile, _ = QFileDialog.getOpenFileName(
                    parent=self,
                    caption="Select NCNN Param",
                    dir=os.path.dirname(modelBinFile),
                    filter=os.path.basename(modelBinFile).replace(".bin", ".param"),
                )
                if modelParamFile == "":
                    RegularQTPopup("Please select a param file!")
                    return
                outputModelFolder = os.path.join(
                    CUSTOM_MODELS_PATH, os.path.basename(modelBinFile).replace(".bin", "")
                )
                createDirectory(outputModelFolder)
                outputBinPath = os.path.join(
                    outputModelFolder, os.path.basename(modelBinFile)
                )
                copyFile(modelBinFile, outputModelFolder)
                outputParamPath = os.path.join(
                    outputModelFolder, os.path.basename(modelParamFile)
                )
                copyFile(modelParamFile, outputModelFolder)

                if os.path.isfile(outputBinPath) and os.path.isfile(outputParamPath):
                    RegularQTPopup(
                        "Model imported successfully!\nPlease restart the app for the changes to take effect."
                    )
                else:
                    RegularQTPopup("Failed to import model!\nPlease try again.")
            else:
                RegularQTPopup(
                    "Custom model does not have a valid\nupscale factor in the name.\nExample: 2x or x2. Skipping import..."
                )

    # output file button
    def openOutputFolder(self):
        """
        Opens a folder,
        sets the directory that is selected to the self.outputFolder variable
        sets the outputFileText to the output directory

        It will also read the input file name, and generate an output file based on it.
        """
        outputFolder = QFileDialog.getExistingDirectory(
            self,
            caption="Select Output Directory",
            dir=self.homeDir,
        )
        self.outputFileText.setText(
            os.path.join(outputFolder, self.setDefaultOutputFile(outputFolder))
        )

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        if reply == QMessageBox.Yes:  # type: ignore
            self.processTab.killRenderProcess()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # setting the pallette

    app.setPalette(Palette())
    window = MainWindow()
    if len(sys.argv) > 1:
        if sys.argv[1] == "--fullscreen":
            window.showFullScreen()
    window.show()
    sys.exit(app.exec())
