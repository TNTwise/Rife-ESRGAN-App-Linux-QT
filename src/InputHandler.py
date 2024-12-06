import cv2


class VideoLoader:
    def __init__(self, inputFile):
        self.inputFile = inputFile

    def loadVideo(self):
        self.capture = cv2.VideoCapture(self.inputFile, cv2.CAP_FFMPEG)

    def isValidVideo(self):
        return self.capture.isOpened()

    def getData(self):
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.bitrate = int(self.capture.get(cv2.CAP_PROP_BITRATE))
        self.videoContainer = self.inputFile.split(".")[-1]
        codec = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        self.codec_str = (
            chr(codec & 0xFF)
            + chr((codec >> 8) & 0xFF)
            + chr((codec >> 16) & 0xFF)
            + chr((codec >> 24) & 0xFF)
        )
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps
        self.capture.release()
