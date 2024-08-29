from .DownloadModels import DownloadModel

"""
Key value pairs of the model name in the GUI
Data inside the tuple:
[0] = file in models directory
[1] = file to download
[2] = upscale times
[3] = arch
"""
ncnnInterpolateModels = {
    "RIFE 4.6 (Fastest Model)": ("rife-v4.6", "rife-v4.6.tar.gz", 1, "rife46"),
    "RIFE 4.7 (Smoothest Model)": ("rife-v4.7", "rife-v4.7.tar.gz", 1, "rife47"),
    "RIFE 4.15": ("rife-v4.15", "rife-v4.15.tar.gz", 1, "rife413"),
    "RIFE 4.18": ("rife-v4.18", "rife-v4.18.tar.gz", 1, "rife413"),
    "RIFE 4.20": ("rife-v4.20", "rife-v4.20.tar.gz", 1, "rife420"),
    "RIFE 4.21": ("rife-v4.21", "rife-v4.21.tar.gz", 1, "rife421"),
    "RIFE 4.22 (Latest General Model)": ("rife-v4.22", "rife-v4.22.tar.gz", 1, "rife421"),
    "RIFE 4.22-lite (Recommended Model)": (
        "rife-v4.22-lite",
        "rife-v4.22-lite.tar.gz",
        1,
        "rife422-lite",
    ),
}
pytorchInterpolateModels = {
    "RIFE 4.6 (Fastest Model)": ("rife4.6.pkl", "rife4.6.pkl", 1, "rife46"),
    "RIFE 4.7 (Smoothest Model)": ("rife4.7.pkl", "rife4.7.pkl", 1, "rife47"),
    "RIFE 4.15": ("rife4.15.pkl", "rife4.15.pkl", 1, "rife413"),
    "RIFE 4.18": ("rife4.18.pkl", "rife4.18.pkl", 1, "rife413"),
    "RIFE 4.20": ("rife4.20.pkl", "rife4.20.pkl", 1, "rife420"),
    "RIFE 4.21": ("rife4.21.pkl", "rife4.21.pkl", 1, "rife421"),
    "RIFE 4.22 (Latest General Model)": ("rife4.22.pkl", "rife4.22.pkl", 1, "rife421"),
    "RIFE 4.22-lite (Recommended Model)": (
        "rife4.22-lite.pkl",
        "rife4.22-lite.pkl",
        1,
        "rife422-lite",
    ),
}
tensorrtInterpolateModels = {
    "RIFE 4.6 (Fastest Model)": ("rife4.6.pkl", "rife4.6.pkl", 1, "rife46"),
    "RIFE 4.7 (Smoothest Model)": ("rife4.7.pkl", "rife4.7.pkl", 1, "rife47"),
    "RIFE 4.15": ("rife4.15.pkl", "rife4.15.pkl", 1, "rife413"),
    "RIFE 4.18": ("rife4.18.pkl", "rife4.18.pkl", 1, "rife413"),
    "RIFE 4.20": ("rife4.20.pkl", "rife4.20.pkl", 1, "rife420"),
    "RIFE 4.21": ("rife4.21.pkl", "rife4.21.pkl", 1, "rife421"),
    "RIFE 4.22 (Latest General Model)": ("rife4.22.pkl", "rife4.22.pkl", 1, "rife421"),
    "RIFE 4.22-lite (Recommended Model)": (
        "rife4.22-lite.pkl",
        "rife4.22-lite.pkl",
        1,
        "rife422-lite",
    ),
}
ncnnUpscaleModels = {
    "SPAN (Animation) (2X)": (
        "2x_ModernSpanimationV2",
        "2x_ModernSpanimationV2.tar.gz",
        2,
        "SPAN",
    ),
    "SPAN (Realistic) (High Quality Source) (4X)": (
        "4xNomos8k_span_otf_weak",
        "4xNomos8k_span_otf_weak.tar.gz",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Medium Quality Source) (4X)": (
        "4xNomos8k_span_otf_medium",
        "4xNomos8k_span_otf_medium.tar.gz",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Low Quality Source) (4X)": (
        "4xNomos8k_span_otf_strong",
        "4xNomos8k_span_otf_strong.tar.gz",
        4,
        "SPAN",
    ),
    "Compact (Realistic) (HD Input) (2X)": (
        "2x_OpenProteus_Compact_i2_70K",
        "2x_OpenProteus_Compact_i2_70K.tar.gz",
        2,
        "Compact",
    ),
}
pytorchUpscaleModels = {
    "SPAN (Animation) (2X)": (
        "2x_ModernSpanimationV2.pth",
        "2x_ModernSpanimationV2.pth",
        2,
        "SPAN",
    ),
    "Sudo Shuffle SPAN (Animation) (2X)": (
        "2xSudoShuffleSPAN.pth",
        "2xSudoShuffleSPAN.pth",
        2,
        "SPAN",
    ),
    "SPAN (Realistic) (High Quality Source) (4X)": (
        "4xNomos8k_span_otf_weak.pth",
        "4xNomos8k_span_otf_weak.pth",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Medium Quality Source) (4X)": (
        "4xNomos8k_span_otf_medium.pth",
        "4xNomos8k_span_otf_medium.pth",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Low Quality Source) (4X)": (
        "4xNomos8k_span_otf_strong.pth",
        "4xNomos8k_span_otf_strong.pth",
        4,
        "SPAN",
    ),
    "Compact (Realistic) (HD Input) (2X)": (
        "2x_OpenProteus_Compact_i2_70K.pth",
        "2x_OpenProteus_Compact_i2_70K.pth",
        2,
        "Compact",
    ),
}
tensorrtUpscaleModels = {
    "SPAN (Animation) (2X)": (
        "2x_ModernSpanimationV2.pth",
        "2x_ModernSpanimationV2.pth",
        2,
        "SPAN",
    ),
    "SPAN (Realistic) (High Quality Source) (4X)": (
        "4xNomos8k_span_otf_weak.pth",
        "4xNomos8k_span_otf_weak.pth",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Medium Quality Source) (4X)": (
        "4xNomos8k_span_otf_medium.pth",
        "4xNomos8k_span_otf_medium.pth",
        4,
        "SPAN",
    ),
    "SPAN (Realistic) (Low Quality Source) (4X)": (
        "4xNomos8k_span_otf_strong.pth",
        "4xNomos8k_span_otf_strong.pth",
        4,
        "SPAN",
    ),
    "Compact (Realistic) (HD Input) (2X)": (
        "2x_OpenProteus_Compact_i2_70K.pth",
        "2x_OpenProteus_Compact_i2_70K.pth",
        2,
        "Compact",
    ),
}
class ModelHandler:
    def __init__(self, backend):
        self.backend = backend


    def downloadModels(self):
        """
        Downloads all the models based on the selected backend.
        Returns:
            None
        """

        match self.backend:
            case "ncnn":
                models=ncnnUpscaleModels | ncnnInterpolateModels
                backend_list = ["ncnn"]
            case "pytorch":
                models=pytorchUpscaleModels | pytorchInterpolateModels
                backend_list = ["pytorch"]
            case "tensorrt":
                models=tensorrtUpscaleModels | tensorrtInterpolateModels
                backend_list = ["tensorrt"]
            case "all":
                backend_list = ["ncnn", "pytorch", "tensorrt"]
        for backend in backend_list:
            for model in models:
                DownloadModel(
                    modelFile=models[model][0],
                    downloadModelFile=models[model][1],
                    backend=backend,
                )