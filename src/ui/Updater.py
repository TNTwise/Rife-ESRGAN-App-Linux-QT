import requests
import re

from .QTcustom import DownloadProgressPopup
from ..constants import BACKEND_PATH, PLATFORM
from ..version import version
from ..Util import FileHandler, networkCheck


class ApplicationUpdater:
    def __init__(self):
        pass

    def check_for_updates(self):
        tag = self.get_latest_version_tag(clean_tag=True)
        return not tag == version and int(tag.replace(".", "")) > int(
            version.replace(".", "")
        )  # returns true if there is a new version

    def download_new_version(self):
        pass

    def remove_old_files(self):
        pass

    def move_new_files(self):
        pass

    def build_download_url(self):
        url = f"https://github.com/tntwise/real-video-enhancer/releases/download/{self.get_latest_version_tag()}/real-video-enhancer-{self.get_latest_version_tag()}.zip"

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


if __name__ == "__main__":
    updater = ApplicationUpdater()
    print(updater.check_for_updates())