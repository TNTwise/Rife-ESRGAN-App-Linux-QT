class BuildFFMpegCommand:
    def __init__(self, encoder, quality):
        self.encoder = encoder
        self.quality = quality
        self.encoderDict = {
            "libx264": "libx264",
            "libx265": "libx265",
            "av1": "libsvtav1",
            "vp9": "libvpx-vp9",
        }
        # get most recent settings
        self.qualityToCRF = {
            "Low": "28",
            "Medium": "23",
            "High": "18",
            "Very High": "15",
        }

    def buildFFmpeg(self):
        return f"-c:v {self.encoderDict[self.encoder]} -crf {self.qualityToCRF[self.quality]}"
