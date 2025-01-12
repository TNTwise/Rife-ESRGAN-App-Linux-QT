import requests
import re
import os

from .QTcustom import DownloadProgressPopup
from ..constants import (
    BACKEND_PATH,
    EXE_NAME,
    LIBS_PATH,
    EXE_PATH,
    PLATFORM,
    TEMP_DOWNLOAD_PATH,
    LIBS_NAME,
    CWD,
)
from ..version import version
from ..Util import FileHandler, networkCheck

def setVersion(version):
    version = version


class ApplicationUpdater:
    def __init__(self, test=False):
        if test:
            setVersion("2.1.0")  # for testing purposes
        self.tag = self.get_latest_version_tag()
        self.clean_tag = self.get_latest_version_tag(clean_tag=True)
        if PLATFORM == "win32":
            platform_name = "Windows"
        elif PLATFORM == "darwin":
            platform_name = "MacOS"
        else:
            platform_name = "Linux"

        self.file_name = f"REAL-Video-Enhancer-{self.clean_tag}-{platform_name}.zip"
        self.download_url = self.build_download_url()

    def check_for_updates(self):
        tag = self.get_latest_version_tag(clean_tag=True)
        return not tag == version and int(tag.replace(".", "")) > int(
            version.replace(".", "")
        )  # returns true if there is a new version

    def download_new_version(self):
        FileHandler.createDirectory(TEMP_DOWNLOAD_PATH)
        full_download_path = os.path.join(TEMP_DOWNLOAD_PATH, self.file_name)
        DownloadProgressPopup(
            link=self.download_url,
            downloadLocation=full_download_path,
            title=f"Downloading {self.tag}",
        )
        FileHandler.unzipFile(
            os.path.join(TEMP_DOWNLOAD_PATH, self.file_name), TEMP_DOWNLOAD_PATH
        )

    def remove_old_files(self):
        FileHandler.removeFolder(BACKEND_PATH)
        FileHandler.removeFolder(LIBS_PATH)
        FileHandler.removeFile(EXE_PATH)

    def move_new_files(self):
        if PLATFORM == "linux":
            folder_to_copy_from = "bin"
        elif PLATFORM == "win32":
            folder_to_copy_from = "REAL-Video-Enhancer"
        else:
            folder_to_copy_from = "REAL-Video-Enhancer"

        FileHandler.moveFolder(
            os.path.join(TEMP_DOWNLOAD_PATH, folder_to_copy_from, "backend"),
            os.path.join(BACKEND_PATH),
        )
        FileHandler.moveFolder(
            os.path.join(TEMP_DOWNLOAD_PATH, folder_to_copy_from, EXE_NAME),
            os.path.join(EXE_PATH),
        )
        FileHandler.moveFolder(
            os.path.join(TEMP_DOWNLOAD_PATH, folder_to_copy_from, LIBS_NAME),
            os.path.join(LIBS_PATH),
        )

    def build_download_url(self):
        url = f"https://github.com/tntwise/real-video-enhancer/releases/download/{self.tag}/{self.file_name}"
        return url

    def get_latest_version_tag(self, clean_tag=False) -> str:
        url = "https://api.github.com/repos/tntwise/real-video-enhancer/releases/latest"
        response = requests.get(url)
        if response.status_code == 200:
            latest_release = response.json()
            tag_name = latest_release["tag_name"]

        else:
            print(f"Failed to fetch latest version: {response.status_code}")
            tag_name = version  # return current version if it failed to get the latest versions

        if clean_tag:
            tag_name = re.sub(r"[^0-9.]", "", tag_name)

        return tag_name

    def install_new_update(self):
        if networkCheck():
            if self.check_for_updates():
                self.download_new_version()
                self.remove_old_files()
                self.move_new_files()
                print("Update complete")


if __name__ == "__main__":
    updater = ApplicationUpdater()
    print(updater.check_for_updates())
    print(updater.build_download_url())