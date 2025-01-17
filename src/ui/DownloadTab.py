from PySide6.QtWidgets import QMainWindow
from .QTcustom import RegularQTPopup, NetworkCheckPopup, addNotificationToButton
from ..DownloadDeps import DownloadDependencies
from ..DownloadModels import DownloadModel
from ..ModelHandler import (
    ncnnInterpolateModels,
    pytorchInterpolateModels,
    ncnnUpscaleModels,
    pytorchUpscaleModels,
)
from .Updater import ApplicationUpdater
from ..constants import IS_FLATPAK, MODELS_PATH


def downloadModelsBasedOnInstalledBackend(installed_backends: list):
    if NetworkCheckPopup():
        for backend in installed_backends:
            match backend:
                case "ncnn":
                    for model in ncnnInterpolateModels:
                        DownloadModel(
                            model, ncnnInterpolateModels[model][1], MODELS_PATH
                        )
                    for model in ncnnUpscaleModels:
                        DownloadModel(model, ncnnUpscaleModels[model][1], MODELS_PATH)
                case "pytorch":  # no need for tensorrt as it uses pytorch models
                    for model in pytorchInterpolateModels:
                        DownloadModel(
                            model, pytorchInterpolateModels[model][1], MODELS_PATH
                        )
                    for model in pytorchUpscaleModels:
                        DownloadModel(
                            model, pytorchUpscaleModels[model][1], MODELS_PATH
                        )


class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
        backends: list,
    ):
        self.parent = parent
        self.downloadDeps = DownloadDependencies()
        self.backends = backends
        self.applicationUpdater = ApplicationUpdater()

        # set this all to not visible, as scrapping the idea for now.

        self.parent.ApplicationUpdateContainer.setVisible(False)
        # elif self.applicationUpdater.check_for_updates():
        #    addNotificationToButton(button=self.parent.UpdateApplicationButton)
        #    addNotificationToButton(button=self.parent.downloadBtn)
        self.QButtonConnect()

    def QButtonConnect(self):
        self.parent.downloadNCNNBtn.clicked.connect(lambda: self.download("ncnn", True))
        self.parent.downloadTorchCUDABtn.clicked.connect(
            lambda: self.download("torch_cuda", True)
        )
        self.parent.downloadTensorRTBtn.clicked.connect(
            lambda: self.download("tensorrt", True)
        )
        self.parent.downloadTorchROCmBtn.clicked.connect(
            lambda: self.download("torch_rocm", True)
        )
        self.parent.downloadDirectMLBtn.clicked.connect(
            lambda: self.download("directml", True)
        )
        self.parent.downloadAllModelsBtn.clicked.connect(
            lambda: downloadModelsBasedOnInstalledBackend(
                ["ncnn", "pytorch", "tensorrt", "directml"]
            )
        )
        self.parent.downloadSomeModelsBasedOnInstalledBackendbtn.clicked.connect(
            lambda: downloadModelsBasedOnInstalledBackend(self.backends)
        )
        self.parent.uninstallNCNNBtn.clicked.connect(
            lambda: self.download("ncnn", False)
        )
        self.parent.uninstallTorchCUDABtn.clicked.connect(
            lambda: self.download("torch_cuda", False)
        )
        self.parent.uninstallTensorRTBtn.clicked.connect(
            lambda: self.download("tensorrt", False)
        )
        self.parent.uninstallTorchROCmBtn.clicked.connect(
            lambda: self.download("torch_rocm", False)
        )
        self.parent.uninstallDirectMLBtn.clicked.connect(
            lambda: self.download("directml", False)
        )
        self.parent.selectPytorchCustomModel.clicked.connect(
            lambda: self.parent.importCustomModel("pytorch")
        )
        self.parent.selectNCNNCustomModel.clicked.connect(
            lambda: self.parent.importCustomModel("ncnn")
        )
        self.parent.UpdateApplicationButton.clicked.connect(
            self.applicationUpdater.install_new_update
        )

    def download(self, dep, install: bool = True):
        """
        Downloads the specified dependency.
        Parameters:
        - dep (str): The name of the dependency to download.
        Returns:
        - None
        """
        if NetworkCheckPopup(
            "https://pypi.org/"
        ):  # check for network before installing
            match dep:
                case "ncnn":
                    self.downloadDeps.downloadNCNNDeps(install)
                case "torch_cuda":
                    self.downloadDeps.downloadPyTorchCUDADeps(install)
                case "tensorrt":
                    self.downloadDeps.downloadTensorRTDeps(install)
                case "torch_rocm":
                    self.downloadDeps.downloadPyTorchROCmDeps(install)
                case "directml":
                    self.downloadDeps.downloadDirectMLDeps(install)
            RegularQTPopup(
                "Download Complete\nPlease restart the application to apply changes."
            )
