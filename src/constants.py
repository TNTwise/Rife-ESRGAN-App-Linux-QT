import os
import sys

PLATFORM = sys.platform  # win32, darwin, linux

IS_FLATPAK = "FLATPAK_ID" in os.environ

CWD = (
    os.path.join(
        os.path.expanduser("~"), ".var", "app", "io.github.tntwise.REAL-Video-Enhancer"
    )
    if IS_FLATPAK
    else os.getcwd()
)

EXE_NAME = "REAL-Video-Enhancer.exe" if PLATFORM == "win32" else "REAL-Video-Enhancer"
LIBS_NAME = "_internal" if PLATFORM == "win32" else "lib"
# dirs
HOME_PATH = os.path.expanduser("~")
MODELS_PATH = os.path.join(CWD, "models")
CUSTOM_MODELS_PATH = os.path.join(CWD, "custom_models")
VIDEOS_PATH = (
    os.path.join(HOME_PATH, "Desktop")
    if PLATFORM == "darwin"
    else os.path.join(HOME_PATH, "Videos")
)
BACKEND_PATH = "/app/bin/backend" if IS_FLATPAK else os.path.join(CWD, "backend")
TEMP_DOWNLOAD_PATH = os.path.join(CWD, "temp")
# exes
FFMPEG_PATH = (
    os.path.join(CWD, "bin", "ffmpeg.exe")
    if PLATFORM == "win32"
    else os.path.join(CWD, "bin", "ffmpeg")
)
PYTHON_PATH = (
    os.path.join(CWD, "python", "python", "python.exe")
    if PLATFORM == "win32"
    else os.path.join(CWD, "python", "python", "bin", "python3")
)
EXE_PATH = os.path.join(
    CWD,
    EXE_NAME,
)
LIBS_PATH = os.path.join(
    CWD,
    LIBS_NAME,
)

# is installed
IS_INSTALLED = os.path.isfile(FFMPEG_PATH) and os.path.isfile(PYTHON_PATH)

IMAGE_SHARED_MEMORY_ID = "/image_preview" + str(os.getpid())
PAUSED_STATE_SHARED_MEMORY_ID = "/paused_state" + str(os.getpid())
INPUT_TEXT_FILE = os.path.join(CWD, f"INPUT{os.getpid()}.txt")
if "--swap-flatpak-checks" in sys.argv:
    IS_FLATPAK = not IS_FLATPAK