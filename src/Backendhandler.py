import os
import sys

from .constants import BACKEND_PATH, PYTHON_PATH, PLATFORM, IS_INSTALLED, IS_FLATPAK
from .Util import (
    log,
    networkCheck,
)
from .version import version


class BackendHandler:
    def __init__(self, parent):
        self.parent = parent

    def enableCorrectBackends(self):
        self.parent.downloadTorchROCmBtn.setEnabled(PLATFORM == "linux")
        if PLATFORM == "darwin":
            self.parent.downloadTorchCUDABtn.setEnabled(False)
            self.parent.downloadTensorRTBtn.setEnabled(False)
        if IS_FLATPAK:
            self.parent.downloadTorchCUDABtn.setEnabled(False)
            self.parent.downloadTorchROCmBtn.setEnabled(False)
            self.parent.downloadTensorRTBtn.setEnabled(False)

        # disable as it is not complete
        try:
            self.parent.downloadDirectMLBtn.setEnabled(False)
            if PLATFORM != "win32":
                self.parent.downloadDirectMLBtn.setEnabled(False)
        except Exception as e:
            print(e)

    def hideUninstallButtons(self):
        self.parent.uninstallTorchCUDABtn.setVisible(False)
        self.parent.uninstallTorchROCmBtn.setVisible(False)
        self.parent.uninstallNCNNBtn.setVisible(False)
        self.parent.uninstallTensorRTBtn.setVisible(False)
        self.parent.uninstallDirectMLBtn.setVisible(False)

    def showUninstallButton(self, backends):
        if "pytorch (cuda)" in backends:
            self.parent.downloadTorchCUDABtn.setVisible(False)
            self.parent.uninstallTorchCUDABtn.setVisible(True)
        if "pytorch (rocm)" in backends:
            self.parent.downloadTorchROCmBtn.setVisible(False)
            self.parent.uninstallTorchROCmBtn.setVisible(True)
        if "ncnn" in backends:
            self.parent.downloadNCNNBtn.setVisible(False)
            self.parent.uninstallNCNNBtn.setVisible(True)
        if "tensorrt" in backends:
            self.parent.downloadTensorRTBtn.setVisible(False)
            self.parent.uninstallTensorRTBtn.setVisible(True)

        # disable as it is not complete
        try:
            self.parent.downloadDirectMLBtn.setEnabled(False)
            if PLATFORM != "win32":
                self.parent.downloadDirectMLBtn.setEnabled(False)
        except Exception as e:
            print(e)

    def setupBackendDeps(self):
        # need pop up window
        from .DownloadDeps import DownloadDependencies

        downloadDependencies = DownloadDependencies()
        downloadDependencies.downloadBackend(version)
        if not IS_INSTALLED:
            from .ui.QTcustom import RegularQTPopup

            if networkCheck():
                # Dont flip these due to shitty code!
                downloadDependencies.downloadFFMpeg()
                downloadDependencies.downloadPython()
                if PLATFORM == "win32":
                    downloadDependencies.downloadVCREDLIST()
            else:
                RegularQTPopup(
                    "Cannot install required dependencies!\nThe first launch of the app requires internet."
                )
                sys.exit()

    def recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(
        self, firstIter=True
    ):
        from .DownloadDeps import DownloadDependencies
        from .ui.QTcustom import RegularQTPopup, DownloadDepsDialog

        """
        will keep trying until the user installs at least 1 backend, happens when user tries to close out of backend slect and gets an error
        """
        try:
            self.availableBackends, self.fullOutput = self.getAvailableBackends()
            if not len(self.availableBackends) == 0:
                return self.availableBackends, self.fullOutput
        except SyntaxError as e:
            log(str(e))
        if not firstIter:
            RegularQTPopup("Please install at least 1 backend!")
        downloadDependencies = DownloadDependencies()
        DownloadDepsDialog(
            ncnnDownloadBtnFunc=lambda: downloadDependencies.downloadNCNNDeps(True),
            pytorchCUDABtnFunc=lambda: downloadDependencies.downloadPyTorchCUDADeps(
                True
            ),
            pytorchROCMBtnFunc=lambda: downloadDependencies.downloadPyTorchROCmDeps(
                True
            ),
            trtBtnFunc=lambda: downloadDependencies.downloadTensorRTDeps(True),
            directmlBtnFunc=lambda: downloadDependencies.downloadDirectMLDeps(True),
        )
        return self.recursivlyCheckIfDepsOnFirstInstallToMakeSureUserHasInstalledAtLeastOneBackend(
            firstIter=False
        )

    def getAvailableBackends(self):
        from .ui.QTcustom import SettingUpBackendPopup

        output = SettingUpBackendPopup(
            [
                PYTHON_PATH,
                "-W",
                "ignore",
                os.path.join(BACKEND_PATH, "rve-backend.py"),
                "--list_backends",
            ]
        )
        output: str = output.getOutput()
        output = output.split(" ")
        # hack to filter out bad find
        new_out = ""
        for word in output:
            if "objc" in word:
                continue
            if "[Torch-TensorRT]" in word:
                continue
            new_out += word + " "

        new_out = new_out.replace(
            "Unable to import quantization op. Please install modelopt library (https://github.com/NVIDIA/TensorRT-Model-Optimizer?tab=readme-ov-file#installation) to add support for compiling quantized models\n",
            "",
        )
        new_out = new_out.replace(
            "WARNING: - Unable to read CUDA capable devices. Return status: 35\n", ""
        )
        # Find the part of the output containing the backends list
        output = new_out
        start = output.find("[")
        end = output.find("]") + 1
        backends_str = output[start:end]
        # Convert the string representation of the list to an actual list
        backends_str = backends_str.replace("[", "").replace("]", "").replace("'", "")
        backends = backends_str.split(",")
        return backends, output
