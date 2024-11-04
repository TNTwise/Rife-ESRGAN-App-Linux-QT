import subprocess
import os
import time
import re
import numpy as np
import cv2
from collections import deque
from .Util import bytesToImg


class NPMeanSCDetect:
    """
    takes in an image as np array and calculates the mean, with ability to use it for scene detect and upscale skip
    """

    def __init__(self, sensitivity: int = 2):
        self.i0 = None
        self.i1 = None
        # multiply sensitivity by 10 for more representative results
        self.sensitivity = sensitivity * 10

    # a simple scene detect based on mean
    def sceneDetect(self, img1):
        if self.i0 is None:
            self.i0 = img1
            self.image0mean = np.mean(self.i0)
            return
        self.i1 = img1
        img1mean = np.mean(self.i1)
        if (
            self.image0mean > img1mean + self.sensitivity
            or self.image0mean < img1mean - self.sensitivity
        ):
            self.image0mean = img1mean
            return True
        self.image0mean = img1mean
        return False


class NPMeanSegmentedSCDetect:
    """
    takes in an image as np array and calculates the mean, with ability to use it for scene detect
    Args:
        sensitivity: int: sensitivity of the scene detect
        segments: int: number of segments to split the image into
        maxDetections: int: number of detections in a segmented scene to trigger a scene change, default is half the segments
    """

    def __init__(
        self, sensitivity: int = 2, segments: int = 10, maxDetections: int = None
    ):
        self.i0 = None
        self.i1 = None
        if maxDetections is None:
            maxDetections = segments // 2 if segments > 1 else 1
        # multiply sensitivity by 10 for more representative results
        self.sensitivity = sensitivity * 10
        self.segments = segments
        self.maxDetections = maxDetections

    def segmentImage(self, img: np.ndarray):
        # split image into segments
        # calculate mean of each segment
        # return list of means
        h, w = img.shape[:2]
        segment_height = h // self.segments
        segment_width = w // self.segments

        means = {}
        for i in range(self.segments):
            for j in range(self.segments):
                segment = img[
                    i * segment_height : (i + 1) * segment_height,
                    j * segment_width : (j + 1) * segment_width,
                ]
                means[i] = np.mean(segment)

        return means

    # a simple scene detect based on mean
    def sceneDetect(self, img1):
        if self.i0 is None:
            self.i0 = img1
            self.segmentsImg1Mean = self.segmentImage(self.i0)
            return
        self.i1 = img1
        segmentsImg2Mean = self.segmentImage(self.i1)
        detections = 0
        for key, value in self.segmentsImg1Mean.items():
            if (
                value > segmentsImg2Mean[key] + self.sensitivity
                or value < segmentsImg2Mean[key] - self.sensitivity
            ):
                self.segmentsImg1Mean = segmentsImg2Mean
                detections += 1
                if detections >= self.maxDetections:
                    return True
        self.segmentsImg1Mean = segmentsImg2Mean
        return False


class NPMeanDiffSCDetect:
    def __init__(self, sensitivity=2):
        self.sensativity = (
            sensitivity * 10
        )  # multiply by 10 for more representative results
        self.i0 = None
        self.i1 = None

    def sceneDetect(self, img1):
        if self.i0 is None:
            self.i0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            return

        self.i1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(self.i1, self.i0)

        mean_diff = np.mean(frame_diff)
        if mean_diff > self.sensativity:
            self.i0 = self.i1
            return True
        self.i0 = self.i1
        return False


class FFMPEGSceneDetect:
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.pipe_name = 'image_pipe'
        self.ffmpeg_process = None
        self.scene_changed = False
        self._create_pipe()
        self._start_ffmpeg()

    def _create_pipe(self):
        """Create a named pipe (FIFO) for FFmpeg."""
        try:
            os.mkfifo(self.pipe_name)
        except FileExistsError:
            pass  # Ignore if the pipe already exists

    def _start_ffmpeg(self):
        """Start the FFmpeg process to read from the named pipe."""
        cmd = [
            'ffmpeg',
            '-f', 'image2pipe',
            '-i', self.pipe_name,
            '-f', 'image2pipe',
            '-i', self.pipe_name,
            '-filter_complex', f"blend=difference,select='gt(scene,{self.threshold})'",
            '-f', 'null',
            '-'
        ]
        self.ffmpeg_process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)

    def sceneDetect(self, img1):
        """Send two images to the FFmpeg process through the named pipe."""
        with open(self.pipe_name, 'wb') as fifo:
            fifo.write(img1)

    def check_scene_change(self):
        """Check if there has been a scene change."""
        if self.ffmpeg_process.poll() is not None:
            return self.scene_changed

        # Read stderr line by line
        while True:
            line = self.ffmpeg_process.stderr.readline()
            if not line:
                break
            # Look for scene change output
            if re.search(r'frame=\s*\d+\s+.*scene', line):
                self.scene_changed = True
                return True  # Scene change detected

        return False  # No scene change detected

    def monitor(self):
        """Continuously monitor for scene changes."""
        try:
            while True:
                self.send_images('image1.jpg', 'image2.jpg')
                if self.check_scene_change():
                    print("Scene change detected!")
                else:
                    print("No scene change.")
                time.sleep(1)  # Adjust as needed
        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
        if os.path.exists(self.pipe_name):
            os.remove(self.pipe_name)

class SceneDetect:
    """
    Class to detect scene changes based on a few parameters
    sceneChangeSsensitivity: This dictates the sensitivity where a scene detect between frames is activated
        - Lower means it is more suseptable to triggering a scene change
        -
    """

    def __init__(
        self,
        sceneChangeMethod: str = "mean",
        sceneChangeSensitivity: float = 2.0,
        width: int = 1920,
        height: int = 1080,
    ):
        self.width = width
        self.height = height
        # this is just the argument from the command line, default is mean
        if sceneChangeMethod == "mean":
            self.detector = NPMeanSCDetect(sensitivity=sceneChangeSensitivity)
        elif sceneChangeMethod == "mean_diff":
            self.detector = NPMeanDiffSCDetect(sensitivity=sceneChangeSensitivity)
        elif sceneChangeMethod == "mean_segmented":
            self.detector = NPMeanSegmentedSCDetect(
                sensitivity=sceneChangeSensitivity, segments=4
            )
        elif sceneChangeMethod == "ffmpeg":
            self.detector = FFMPEGSceneDetect(
                threshold=sceneChangeSensitivity / 10,
            )
        else:
            raise ValueError("Invalid scene change method")

    def detect(self, frame):
        frame = bytesToImg(frame, width=self.width, height=self.height)
        out = self.detector.sceneDetect(frame)
        return out
