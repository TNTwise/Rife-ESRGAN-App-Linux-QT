import os

def checkForCUDA() -> bool:
    try:
        import torch
        import torchvision
        import cupy
    except Exception as e:
        return False
    if cupy.cuda.get_cuda_path() == None:
        return False
    return True

__version__ = "2.1.5"
IS_FLATPAK = "FLATPAK_ID" in os.environ

if IS_FLATPAK:
    CWD = os.path.join(
        os.path.expanduser("~"), ".var", "app", "io.github.tntwise.REAL-Video-Enhancer"
    )
    if not os.path.exists(CWD):
        CWD = os.path.join(
            os.path.expanduser("~"),
            ".var",
            "app",
            "io.github.tntwise.REAL-Video-EnhancerV2",
        )
else:
    CWD = os.getcwd()

FFMPEG_PATH = os.path.join(CWD, "bin", "ffmpeg")
FFMPEG_LOG_FILE = os.path.join(CWD, "ffmpeg_log.txt")
MODELS_DIRECTORY = os.path.join(CWD, "models")
HAS_SYSTEM_CUDA = checkForCUDA()
