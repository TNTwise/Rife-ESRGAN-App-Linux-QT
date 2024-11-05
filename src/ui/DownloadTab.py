from PySide6.QtWidgets import QMainWindow
from .QTcustom import RegularQTPopup, NetworkCheckPopup
from ..DownloadDeps import DownloadDependencies
from ..ModelHandler import downloadModelsBasedOnInstalledBackend


class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
        installed_backends: list,
    ):
        self.parent = parent
        self.installed_backends = installed_backends
        self.downloadDeps = DownloadDependencies()
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
