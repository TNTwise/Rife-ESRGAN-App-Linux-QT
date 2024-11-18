import os
import re

from .Util import createDirectory, printAndLog
from .constants import CUSTOM_MODELS_PATH

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
    "RIFE 4.18 (Recommended for realistic scenes)": (
        "rife-v4.18",
        "rife-v4.18.tar.gz",
        1,
        "rife413",
    ),
    "RIFE 4.22 (Slowest Model, Animation)": (
        "rife-v4.22",
        "rife-v4.22.tar.gz",
        1,
        "rife421",
    ),
    "RIFE 4.22-lite (Latest LITE model)": (
        "rife-v4.22-lite",
        "rife-v4.22-lite.tar.gz",
        1,
        "rife422-lite",
    ),
    "RIFE 4.25 (Latest General Model, Recommended)": (
        "rife-v4.25",
        "rife-v4.25.tar.gz",
        1,
        "rife425",
    ),
}
pytorchInterpolateModels = {
    "GMFSS (Slowest Model, Animation)": ("GMFSS.pkl", "GMFSS.pkl", 1, "gmfss"),
    "RIFE 4.6 (Fastest Model)": ("rife4.6.pkl", "rife4.6.pkl", 1, "rife46"),
    "RIFE 4.7 (Smoothest Model)": ("rife4.7.pkl", "rife4.7.pkl", 1, "rife47"),
    "RIFE 4.15": ("rife4.15.pkl", "rife4.15.pkl", 1, "rife413"),
    "RIFE 4.18 (Recommended for realistic scenes)": (
        "rife4.18.pkl",
        "rife4.18.pkl",
        1,
        "rife413",
    ),
    "RIFE 4.22 (Slowest Model, Animation)": (
        "rife4.22.pkl",
        "rife4.22.pkl",
        1,
        "rife421",
    ),
    "RIFE 4.22-lite (Latest LITE model)": (
        "rife4.22-lite.pkl",
        "rife4.22-lite.pkl",
        1,
        "rife422-lite",
    ),
    "RIFE 4.25 (Latest General Model, Recommended)": (
        "rife4.25.pkl",
        "rife4.25.pkl",
        1,
        "rife425",
    ),
}
tensorrtInterpolateModels = {
    "RIFE 4.6 (Fastest Model)": ("rife4.6.pkl", "rife4.6.pkl", 1, "rife46"),
    "RIFE 4.7 (Smoothest Model)": ("rife4.7.pkl", "rife4.7.pkl", 1, "rife47"),
    "RIFE 4.15": ("rife4.15.pkl", "rife4.15.pkl", 1, "rife413"),
    "RIFE 4.18 (Recommended for realistic scenes)": (
        "rife4.18.pkl",
        "rife4.18.pkl",
        1,
        "rife413",
    ),
    "RIFE 4.22 (Slowest Model, Animation)": (
        "rife4.22.pkl",
        "rife4.22.pkl",
        1,
        "rife421",
    ),
    "RIFE 4.22-lite (Latest LITE model)": (
        "rife4.22-lite.pkl",
        "rife4.22-lite.pkl",
        1,
        "rife422-lite",
    ),
    "RIFE 4.25 (Latest General Model, Recommended)": (
        "rife4.25.pkl",
        "rife4.25.pkl",
        1,
        "rife425",
    ),
}
ncnnUpscaleModels = {
    
    "SPAN Spanimation (Animation) (2X) (Fast)": (
        "2x_ModernSpanimationV2",
        "2x_ModernSpanimationV2.tar.gz",
        2,
        "SPAN",
    ),
    "SPAN Nomos8k (Realistic) (High Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_weak",
        "4xNomos8k_span_otf_weak.tar.gz",
        4,
        "SPAN",
    ),
    "SPAN Nomos8k (Realistic) (Medium Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_medium",
        "4xNomos8k_span_otf_medium.tar.gz",
        4,
        "SPAN",
    ),
    "SPAN Nomos8k (Realistic) (Low Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_strong",
        "4xNomos8k_span_otf_strong.tar.gz",
        4,
        "SPAN",
    ),
    "RealCUGAN Pro (Animation) (2X) (Slow)": (
        "up2x-conservative",
        "up2x-conservative.tar.gz",
        2,
        "compact",
    ),
    "RealCUGAN Pro (Animation) (3X) (Slow)": (
        "up3x-conservative",
        "up2x-conservative.tar.gz",
        3,
        "compact",
    ),
    "RealESRGAN OpenProteus (Realistic) (HD Input) (2X) (Fast)": (
        "2x_OpenProteus_Compact_i2_70K",
        "2x_OpenProteus_Compact_i2_70K.tar.gz",
        2,
        "Compact",
    ),
    "RealESRGAN JaNai (Animation) (2X) (Fast)": (
        "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k",
        "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.tar.gz",
        2,
        "Compact",
    ),
    "RealESRGAN AnimeVideo V3 (Animation) (2X) (Fast)": (
        "realesr-animevideov3-x2",
        "realesr-animevideov3-x2.tar.gz",
        2,
        "compact",
    ),
    "RealESRGAN AnimeVideo V3 (Animation) (3X) (Fast)": (
        "realesr-animevideov3-x3",
        "realesr-animevideov3-x3.tar.gz",
        3,
        "compact",
    ),
    "RealESRGAN AnimeVideo V3 (Animation) (4X) (Fast)": (
        "realesr-animevideov3-x4",
        "realesr-animevideov3-x4.tar.gz",
        4,
        "compact",
    ),
    "RealESRGAN Plus (General Model) (4X) (Slow)": (
    "realesrgan-x4plus",
    "realesrgan-x4plus.tar.gz",
    4,
    "esrgan",
),
"RealESRGAN Plus (Animation Model) (4X) (Slow)": (
    "realesrgan-x4plus-anime",
    "realesrgan-x4plus-anime.tar.gz",
    4,
    "esrgan",
),
}

pytorchUpscaleModels = {
    "SPAN Spanimation (Animation) (2X) (Fast)": (
        "2x_ModernSpanimationV2.pth",
        "2x_ModernSpanimationV2.pth",
        2,
        "SPAN",
    ),
    "SPAN Nomos8k (Realistic) (High Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_weak.pth",
        "4xNomos8k_span_otf_weak.pth",
        4,
        "SPAN",
    ),
    "SPAN Nomos8k (Realistic) (Medium Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_medium.pth",
        "4xNomos8k_span_otf_medium.pth",
        4,
        "SPAN",
    ),
    "SPAN Nomos8k (Realistic) (Low Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_strong.pth",
        "4xNomos8k_span_otf_strong.pth",
        4,
        "SPAN",
    ),
    "RealESRGAN OpenProteus (Realistic) (HD Input) (2X) (Fast)": (
        "2x_OpenProteus_Compact_i2_70K.pth",
        "2x_OpenProteus_Compact_i2_70K.pth",
        2,
        "Compact",
    ),
    "RealESRGAN JaNai (Animation) (2X) (Fast)": (
        "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth",
        "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth",
        2,
        "Compact",
    ),
}
pytorchDenoiseModels = {
    "SCUNet Color (1x) (Slow)": (
        "scunet_color_real_psnr.pth",
        "scunet_color_real_psnr.pth",
        1,
        "scunet",
    )
}
"""
    "Sudo Shuffle SPAN (Animation) (2X) (Fast)": (
        "2xSudoShuffleSPAN.pth",
        "2xSudoShuffleSPAN.pth",
        2,
        "SPAN",
    ),
"""
tensorrtUpscaleModels = {
    "SPAN Spanimation (Animation) (2X) (Fast)": (
        "2x_ModernSpanimationV2.pth",
        "2x_ModernSpanimationV2.pth",
        2,
        "SPAN",
    ),
    "SPAN Nomos8k (Realistic) (High Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_weak.pth",
        "4xNomos8k_span_otf_weak.pth",
        4,
        "SPAN",
    ),
    "SPAN Nomos8k (Realistic) (Medium Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_medium.pth",
        "4xNomos8k_span_otf_medium.pth",
        4,
        "SPAN",
    ),
    "SPAN Nomos8k (Realistic) (Low Quality Source) (4X) (Fast)": (
        "4xNomos8k_span_otf_strong.pth",
        "4xNomos8k_span_otf_strong.pth",
        4,
        "SPAN",
    ),
    "RealESRGAN OpenProteus (Realistic) (HD Input) (2X) (Fast)": (
        "2x_OpenProteus_Compact_i2_70K.pth",
        "2x_OpenProteus_Compact_i2_70K.pth",
        2,
        "Compact",
    ),
    "RealESRGAN JaNai (Animation) (2X) (Fast)": (
        "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth",
        "2x_AnimeJaNai_HD_V3_Sharp1_Compact_430k.pth",
        2,
        "Compact",
    ),
}
onnxInterpolateModels = {
    "RIFE 4.22 (Recommended Model)": (
        "rife422_v2_ensembleFalse_op20_clamp.onnx",
        "rife422_v2_ensembleFalse_op20_clamp.onnx",
        1,
        "rife422-lite",
    ),
}
onnxUpscaleModels = {
    "SPAN (Animation) (2X)": (
        "2x_ModernSpanimationV2_clamp_op20.onnx",
        "2x_ModernSpanimationV2_clamp_op20.onnx",
        2,
        "SPAN",
    ),
}
def getCustomModelScale(model):
    pattern = r"\d+x|x+\d"
    matches = re.findall(pattern, model)
    if len(matches) > 0:
        upscaleFactor = int(matches[0].replace("x", ""))  
        return upscaleFactor
    return None
# detect custom models
createDirectory(CUSTOM_MODELS_PATH)
customPytorchUpscaleModels = {}
customNCNNUpscaleModels = {}
for model in os.listdir(CUSTOM_MODELS_PATH):

    upscaleFactor = getCustomModelScale(model)
    if upscaleFactor:
        model_path = os.path.join(CUSTOM_MODELS_PATH, model)
        if os.path.exists(model_path):
            if not os.path.isfile(model_path):
                customNCNNUpscaleModels[model] = (model, model, upscaleFactor, "custom")
        if model.endswith(".pth"):
            customPytorchUpscaleModels[model] = (model, model, upscaleFactor, "custom")
    else:
        printAndLog(
            f"Custom model {model} does not have a valid upscale factor in the name, example: 2x or x2. Skipping import..."
        )
    
pytorchUpscaleModels = pytorchUpscaleModels | customPytorchUpscaleModels
tensorrtUpscaleModels = tensorrtUpscaleModels | customPytorchUpscaleModels
ncnnUpscaleModels = ncnnUpscaleModels | customNCNNUpscaleModels
totalModels = (
    onnxInterpolateModels
    | onnxUpscaleModels
    | pytorchInterpolateModels
    | pytorchUpscaleModels
    | pytorchDenoiseModels
    | ncnnInterpolateModels
    | ncnnUpscaleModels
    | tensorrtInterpolateModels
    | tensorrtUpscaleModels
)  # this doesnt include all models due to overwriting, but includes every case of every unique model name



