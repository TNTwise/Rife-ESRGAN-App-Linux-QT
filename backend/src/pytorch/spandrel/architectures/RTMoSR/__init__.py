import math

from typing_extensions import override

from ...util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    StateDict,
)
from .__arch.RTMoSR import RTMoSR


class RTMoSRArch(Architecture[RTMoSR]):
    def __init__(self):
        super().__init__(
            id="RTMoSR",
            name="Real Time MoSR",
            detect=KeyCondition.has_all(
                "body.0.norm.weight",
                "body.0.norm.bias",
                "body.0.fc1.alpha",
                "body.0.fc1.conv1.weight",
                "body.0.fc1.conv1.bias",
                "body.0.fc1.conv2.weight",
                "body.0.fc1.conv2.bias",
                "body.0.fc1.conv3.sk.weight",
                "body.0.fc1.conv3.sk.bias",
                "body.0.fc1.conv3.conv.0.weight",
                "body.0.fc1.conv3.conv.0.bias",
                "body.0.fc1.conv3.conv.1.weight",
                "body.0.fc1.conv3.conv.1.bias",
                "body.0.fc1.conv3.conv.2.weight",
                "body.0.fc1.conv3.conv.2.bias",
                "body.0.fc1.conv3.eval_conv.weight",
                "body.0.fc1.conv3.eval_conv.bias",
                "body.0.fc1.conv_3x3_rep.weight",
                "body.0.fc1.conv_3x3_rep.bias",
                "body.0.conv.dwconv_hw.weight",
                "body.0.conv.dwconv_hw.bias",
                "body.0.conv.dwconv_w.weight",
                "body.0.conv.dwconv_w.bias",
                "body.0.conv.dwconv_h.weight",
                "body.0.conv.dwconv_h.bias",
                "body.0.fc2.weight",
                "body.0.fc2.bias",
                "to_img.0.weight",
                "to_img.0.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[RTMoSR]:
        in_nc = 3
        out_nc = 3
        unshuffle = False
        if "to_feat.1.weight" in state_dict:
            unshuffle = True
            scale = math.isqrt(state_dict["to_feat.1.weight"].shape[1] // 3)
            dim = state_dict["to_feat.1.weight"].shape[0]
        else:
            scale = math.isqrt(state_dict["to_img.0.weight"].shape[0] // 3)
            dim = state_dict["to_feat.weight"].shape[0]
        ffn = state_dict["body.0.fc1.conv1.weight"].shape[0] / dim / 2
        n_blocks = get_seq_len(state_dict, "body")

        model = RTMoSR(
            scale=scale,
            dim=dim,
            ffn_expansion=ffn,
            n_blocks=n_blocks,
            unshuffle_mod=unshuffle,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{dim}nf", f"{n_blocks}nc"],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=in_nc,
            output_channels=out_nc,
        )


__all__ = ["RTMoSRArch", "RTMoSR"]
