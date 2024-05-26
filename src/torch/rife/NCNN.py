from rife_ncnn_vulkan_python import Rife
from src.programData.thisdir import thisdir as ts
import os
import numpy as np

thisdir = ts()


class RifeNCNN:
    def __init__(
        self,
        interpolation_factor,
        interpolate_method,
        width,
        height,
        ensemble,
        half,
        threads=2,
        ncnn_gpu=0,
    ):
        self.i0=None
        self.interpolation_factor = interpolation_factor
        self.interpolation_method = interpolate_method
        self.width = width
        self.height = height
        self.ensemble = ensemble
        self.half = half
        self.handleModel()
        self.createInterpolation(ncnn_gpu=ncnn_gpu, threads=threads)
    def cacheFrame(self):
        self.i0 = self.i1.copy()
    def clearCache(self):
        """
        Clears cache when scene change is detected.
        
        Overwrites frame
        """
        self.i0 = None
    def handleModel(self):
        self.modelPath = os.path.join(
            thisdir, "models", "rife", self.interpolation_method
        )

    def createInterpolation(self, ncnn_gpu=0, threads=2):
        self.render = Rife(
            gpuid=ncnn_gpu, num_threads=threads, model=self.modelPath, uhd_mode=False
        )

    def bytesToNpArray(self, bytes):
        return np.ascontiguousarray(
            np.frombuffer(bytes, dtype=np.uint8).reshape(self.height, self.width, 3)
        )

    def run(self, i1):
        if self.i0 is None:
            print('Uncached Frame!')
            self.i0 = self.bytesToNpArray(i1)
            return False
            
        self.i1 = self.bytesToNpArray(i1)
        return True

    def make_inference(self, n):
        output =  np.ascontiguousarray(
            self.render.process_cv2(self.i0, self.i1, timestep=n)
        )
        self.cacheFrame()
        return output
