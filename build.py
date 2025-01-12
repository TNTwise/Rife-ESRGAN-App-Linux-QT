import os
import subprocess
import sys
import os
import subprocess
import sys
import shutil

import urllib.request

linux_and_mac_py_ver = "python3.10"


def checkIfExeExists(exe):
    path = shutil.which(exe)
    return path is not None


def getPlatform():
    return sys.platform


def python_path():
    return (
        "venv\\Scripts\\python.exe" if getPlatform() == "win32" else "venv/bin/python3"
    )


python_version = (
    linux_and_mac_py_ver
    if getPlatform() != "win32" and checkIfExeExists(linux_and_mac_py_ver)
    else "python3"
)


def get_site_packages():
    command = [
        python_path(),
        "-c",
        'import site; print("\\n".join(site.getsitepackages()))',
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    site_packages = result.stdout.strip()
    return site_packages


def download_file(url, destination):
    print(f"Downloading file from {url}")
    urllib.request.urlretrieve(url, destination)
    print("File downloaded successfully")

def zero_mainwindow_size():
    import xml.etree.ElementTree as ET

    def set_mainwindow_size_zero(path="testRVEInterface.ui"):
        tree = ET.parse(path)
        root = tree.getroot()

        geometry = root.find('.//property[@name="geometry"]/rect')
        if geometry is not None:
            width = geometry.find("width")
            height = geometry.find("height")
            if width is not None:
                width.text = "0"
            if height is not None:
                height.text = "0"
            tree.write(path)

    set_mainwindow_size_zero()


def build_gui():
    print("Building GUI")
    zero_mainwindow_size()
    if getPlatform() == "darwin" or getPlatform() == "linux":
        os.system(
            f"{get_site_packages()}/PySide6/Qt/libexec/uic -g python testRVEInterface.ui > mainwindow.py"
        )
    if getPlatform() == "win32":
        os.system(
            r".\venv\Lib\site-packages\PySide6\uic.exe -g python testRVEInterface.ui > mainwindow.py"
        )


def install_pip():
    download_file("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
    command = ["python3", "get-pip.py"]
    subprocess.run(command)


def install_pip_in_venv():
    command = [
        "venv\\Scripts\\python.exe" if getPlatform() == "win32" else "venv/bin/python3",
        "get-pip.py",
    ]
    subprocess.run(command)


def build_resources():
    print("Building resources.rc")
    if getPlatform() == "darwin" or getPlatform() == "linux":
        os.system(
            f"{get_site_packages()}/PySide6/Qt/libexec/rcc -g python resources.qrc > resources_rc.py"
        )
    if getPlatform() == "win32":
        os.system(
            r".\venv\Lib\site-packages\PySide6\rcc.exe -g python resources.qrc > resources_rc.py"
        )


def create_venv():
    print("Creating virtual environment")
    command = [python_version, "-m", "venv", "venv"]
    subprocess.run(command)


def install_requirements_in_venv():
    print("Installing requirements in virtual environment")
    command = [
        python_path(),
        "-m",
        "pip",
        "install",
        "-r",
        "requirements.txt",
    ]

    subprocess.run(command)


def build_executable(dist_dir=None):
    print("Building executable")
    if getPlatform() == "win32" or getPlatform() == "darwin":
        if dist_dir is None:
            dist_dir = "dist"
        command = [
            python_path(),
            "-m",
            "PyInstaller",
            "REAL-Video-Enhancer.py",
            "--collect-all",
            "PySide6",
            "--icon=icons/logo-v2.ico",
            "--noconfirm",
            "--noupx",
            "--distpath",
            dist_dir,
            # "--noconsole", this caused issues, maybe I can fix it later
        ]
    else:
        if dist_dir is None:
            dist_dir = "bin"
        command = [
            python_path(),
            "-m",
            "cx_Freeze",
            "REAL-Video-Enhancer.py",
            "--target-dir",
            dist_dir,
        ]
    subprocess.run(command)


def copy_backend(build_dir=None):
    print("Copying backend")
    if getPlatform() == "win32":
        if build_dir is None:
            build_dir = "dist"
        try:
            os.system(f"cp -r backend {build_dir}/REAL-Video-Enhancer/")
        except Exception:
            pass
        if not os.path.exists(rf"{build_dir}\\REAL-Video-Enhancer\\backend"):
            os.system(f'xcopy "./backend" "./{build_dir}/REAL-Video-Enhancer/" /E /I')
    if getPlatform() == "linux":
        if build_dir is None:
            build_dir = "bin"
        os.system(f"cp -r backend {build_dir}/")


def clean():
    print("Cleaning up")
    os.remove("get-pip.py")


def build_venv():
    create_venv()
    install_pip_in_venv()
    install_requirements_in_venv()


if len(sys.argv) > 1:
    if sys.argv[1] == "--create_venv" or sys.argv[1] == "--build_exe":
        build_venv()

if not os.path.exists("venv"):
    build_venv()

build_gui()
build_resources()

if "--build_dir_override" in sys.argv:
    build_dir = sys.argv[sys.argv.index("--build_dir_override") + 1]
    build_executable(build_dir)
    copy_backend(build_dir=build_dir)
if "--build_exe" in sys.argv and "--build_dir_override" not in sys.argv:
    build_executable()
    copy_backend()
