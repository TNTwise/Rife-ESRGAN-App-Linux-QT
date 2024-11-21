from PySide6.QtWidgets import QMainWindow
from .QTcustom import RegularQTPopup, NetworkCheckPopup
from ..DownloadDeps import DownloadDependencies
from ..DownloadModels import DownloadModel
from ..ModelHandler import ncnnInterpolateModels, pytorchInterpolateModels, ncnnUpscaleModels, pytorchUpscaleModels 
def downloadModelsBasedOnInstalledBackend(installed_backends: list):
    if NetworkCheckPopup():
        for backend in installed_backends:
            match backend:
                case "ncnn":
                    for model in ncnnInterpolateModels:
                        DownloadModel(model, ncnnInterpolateModels[model][1], "ncnn")
                    for model in ncnnUpscaleModels:
                        DownloadModel(model, ncnnUpscaleModels[model][1], "ncnn")
                case "pytorch":  # no need for tensorrt as it uses pytorch models
                    for model in pytorchInterpolateModels:
                        DownloadModel(
                            model, pytorchInterpolateModels[model][1], "pytorch"
                        )
                    for model in pytorchUpscaleModels:
                        DownloadModel(model, pytorchUpscaleModels[model][1], "pytorch")
        """case "directml":
            for model in onnxInterpolateModels:
                DownloadModel(model, onnxInterpolateModels[model][1], "onnx")
            for model in onnxUpscaleModels:
                DownloadModel(model, onnxUpscaleModels[model][1], "onnx")"""

class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
    ):
        self.parent = parent
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
        self.parent.selectNCNNCustomModel.clicked.connect(
            lambda: self.parent.importCustomModel("ncnn")
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
