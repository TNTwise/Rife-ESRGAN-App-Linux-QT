from math import comb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .FeatureNet import FeatureNet
from .gmflow.gmflow import GMFlow
from .IFNet_HDv3 import IFNet
from .MetricNet import MetricNet
from ....constants import HAS_SYSTEM_CUDA
from ..DetectInterpolateArch import ArchDetect
if HAS_SYSTEM_CUDA:
    from ..util.softsplat_cupy import softsplat as warp
else:
    from ..util.softsplat_torch import softsplat as warp



class GMFSS(nn.Module):
    def __init__(
        self,
        model_path,
        model_type: str = "union",
        scale: float = 1.,
        ensemble: bool = False,
        width: int = 1920,
        height: int = 1080,
    ):
        super(GMFSS, self).__init__()
        from .FusionNet_u import GridNet

        combined_state_dict = torch.load(model_path, map_location="cpu")

        archDetect = ArchDetect(combined_state_dict["rife"])
        rife_version = archDetect.getArchName()
        #print(rife_version)
        if rife_version.lower() == "rife46":
            from .IFNet_HDv3 import IFNet
        else:
            # this is dumb, it detects rife4.7 with a stupid hack, so we need to just force load 422
            from .IFNet_HDv3_422 import IFNet
        
        # get gmfss from here, as its a combination of all the models https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/GMFSS.pkl
        self.width = width
        self.height = height
        self.ifnet = IFNet(ensemble=ensemble)
        self.flownet = GMFlow()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()
        
        
        if model_type != "base":
            self.ifnet.load_state_dict(combined_state_dict["rife"])
        self.flownet.load_state_dict(combined_state_dict["flownet"])
        self.metricnet.load_state_dict(combined_state_dict["metricnet"])
        self.feat_ext.load_state_dict(combined_state_dict["feat_ext"])
        self.fusionnet.load_state_dict(combined_state_dict["fusionnet"])
        

        self.model_type = model_type
        self.scale = scale

    def reuse(self, img0, img1):
        feat11, feat12, feat13 = self.feat_ext(img0)
        feat21, feat22, feat23 = self.feat_ext(img1)
        feat_ext0 = [feat11, feat12, feat13]
        feat_ext1 = [feat21, feat22, feat23]

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear")
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear")

        if self.scale != 1.0:
            imgf0 = F.interpolate(img0, scale_factor=self.scale, mode="bilinear")
            imgf1 = F.interpolate(img1, scale_factor=self.scale, mode="bilinear")
        else:
            imgf0 = img0
            imgf1 = img1
        flow01 = self.flownet(imgf0, imgf1)
        flow10 = self.flownet(imgf1, imgf0)
        if self.scale != 1.0:
            flow01 = (
                F.interpolate(flow01, scale_factor=1.0 / self.scale, mode="bilinear")
                / self.scale
            )
            flow10 = (
                F.interpolate(flow10, scale_factor=1.0 / self.scale, mode="bilinear")
                / self.scale
            )

        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)

        return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1

    def forward(self, img0, img1, timestep, scale=None):
        if scale is not None:
            self.scale = scale
        dtype = img0.dtype
        reuse_things = self.reuse(img0, img1)
        flow01, metric0, feat11, feat12, feat13 = (
            reuse_things[0],
            reuse_things[2],
            reuse_things[4][0],
            reuse_things[4][1],
            reuse_things[4][2],
        )
        flow10, metric1, feat21, feat22, feat23 = (
            reuse_things[1],
            reuse_things[3],
            reuse_things[5][0],
            reuse_things[5][1],
            reuse_things[5][2],
        )

        F1t = timestep * flow01
        F2t = (1 - timestep) * flow10

        Z1t = timestep * metric0
        Z2t = (1 - timestep) * metric1

        img0 = F.interpolate(img0, scale_factor=0.5, mode="bilinear")
        I1t = warp(img0, F1t, Z1t, strMode="soft")
        img1 = F.interpolate(img1, scale_factor=0.5, mode="bilinear")
        I2t = warp(img1, F2t, Z2t, strMode="soft")

        if self.model_type == "union":
            rife = self.ifnet(img0, img1, timestep)

        feat1t1 = warp(feat11, F1t, Z1t, strMode="soft")
        feat2t1 = warp(feat21, F2t, Z2t, strMode="soft")

        F1td = F.interpolate(F1t, scale_factor=0.5, mode="bilinear") * 0.5
        Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear")
        feat1t2 = warp(feat12, F1td, Z1d, strMode="soft")
        F2td = F.interpolate(F2t, scale_factor=0.5, mode="bilinear") * 0.5
        Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear")
        feat2t2 = warp(feat22, F2td, Z2d, strMode="soft")

        F1tdd = F.interpolate(F1t, scale_factor=0.25, mode="bilinear") * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor=0.25, mode="bilinear")
        feat1t3 = warp(feat13, F1tdd, Z1dd, strMode="soft")
        F2tdd = F.interpolate(F2t, scale_factor=0.25, mode="bilinear") * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor=0.25, mode="bilinear")
        feat2t3 = warp(feat23, F2tdd, Z2dd, strMode="soft")

        out = self.fusionnet(
            torch.cat(
                [img0, I1t, I2t, img1]
                if self.model_type == "base"
                else [I1t, rife, I2t],
                dim=1,
            ),
            torch.cat([feat1t1, feat2t1], dim=1),
            torch.cat([feat1t2, feat2t2], dim=1),
            torch.cat([feat1t3, feat2t3], dim=1),
        )
        out = out[:, :, : self.height, : self.width]
        return torch.clamp(out, 0, 1)
