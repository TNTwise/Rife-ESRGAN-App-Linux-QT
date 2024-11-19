import yt_dlp
import validators
import os
import cv2

def isYoutubeVideo(url):
    return validators.url(url) and "youtube.com" in url or "youtu.be" in url


class VideoLoader:

    def __init__(self, inputFile):
        self.inputFile = inputFile

    def loadVideo(self):
        self.capture = cv2.VideoCapture(self.inputFile, cv2.CAP_FFMPEG)

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

    def getVideoTitle(self)-> str:
        return self.inputFile


    def releaseCapture(self):
        self.capture.release()  

class YouTubeVideoLoader:

    def __init__(self, inputLink):
        self.inputFile = inputLink
        
    def loadVideo(self):
        self.infoDict = self.getInfoDict()
    
    def getStreams(self):
        return [item['format'] for item in self.infoDict['formats']] # this gets the available streams based on resolution

    def getInfoDict(self):
        ydl_opts = {
            "quiet": True,  # Suppress output
            "noplaylist": True,  # Only check single video, not playlists
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                # Extract info about the video
                info_dict = ydl.extract_info(self.inputFile, download=False)
                # Check if there are available formats
                if info_dict.get("formats"):
                    return info_dict  # Video is downloadable
                else:
                    return False  # No formats available
            except Exception as e:
                print(f"Error occurred: {e}")
                return False


    def checkValidVideo(self):
        return self.infoDict is not False # info dict will be false if no formats are available

    def getVideoContainer(self):
        return self.infoDict['ext']
    
    def getVideoRes(self) -> list[int, int]:
        width = self.infoDict['width']
        height = self.infoDict['height']
        resolution = [width, height]
        return resolution

    def getVideoBitrate(self) -> int:
        return self.infoDict['vbr']


    def getVideoEncoder(self):
        return self.infoDict['vcodec']
    
    def getVideoFPS(self) -> float:
        return self.infoDict['fps']
    
    def getVideoLength(self) -> int:
        
        return self.infoDict['duration']
    
    def getVideoFrameCount(self) -> int:
        
        return self.infoDict['fps'] * self.infoDict['duration']

    def getVideoTitle(self):
        return self.infoDict['title']

    def releaseCapture(self):
        return

    def downloadVideo(self):
        ydl_opts = {
            "quiet": True,  # Suppress output
            "noplaylist": True,  # Only check single video, not playlists
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.inputFile])

    def downloadAudio(self):
        ydl_opts = {
            "quiet": True,  # Suppress output
            "noplaylist": True,  # Only check single video, not playlists
            "format": "bestaudio"
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.inputFile])