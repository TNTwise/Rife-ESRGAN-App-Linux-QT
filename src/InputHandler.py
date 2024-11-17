import yt_dlp
import validators
import os
import cv2

class VideoLoader:

    def __init__(self, inputFile):
        self.inputFile = inputFile
        self.capture = cv2.VideoCapture(inputFile, cv2.CAP_FFMPEG)

    def checkValidVideo(self):
        return self.capture.isOpened()

    def getVideoContainer(self):
        return os.path.splitext(self.inputFile)[1]
    
    def getVideoRes(self) -> list[int, int]:
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = [width, height]
        return resolution

    def getVideoBitrate(self) -> int:
        bitrate = int(self.capture.get(cv2.CAP_PROP_BITRATE))
        return bitrate


    def getVideoEncoder(self):
        codec = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        codec_str = (
            chr(codec & 0xFF)
            + chr((codec >> 8) & 0xFF)
            + chr((codec >> 16) & 0xFF)
            + chr((codec >> 24) & 0xFF)
        )
        return codec_str
    
    def getVideoFPS(self) -> float:
        return self.capture.get(cv2.CAP_PROP_FPS)
    
    def getVideoLength(self) -> int:
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        return duration
    
    def getVideoFrameCount(self) -> int:
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames

    def releaseCapture(self):
        self.capture.release()  

class VideoInputHandler(VideoLoader):
    def __init__(self, inputText):
        self.inputText = inputText
        super().__init__(inputText)

    def isYoutubeLink(self):
        url = self.inputText
        return validators.url(url) and "youtube.com" in url or "youtu.be" in url


    def isValidYoutubeLink(self):
        ydl_opts = {
            "quiet": True,  # Suppress output
            "noplaylist": True,  # Only check single video, not playlists
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Extract info about the video
                info_dict = ydl.extract_info(self.inputText, download=False)
                # Check if there are available formats
                if info_dict.get("formats"):
                    return True  # Video is downloadable
                else:
                    return False  # No formats available
            except Exception as e:
                print(f"Error occurred: {e}")
                return False
