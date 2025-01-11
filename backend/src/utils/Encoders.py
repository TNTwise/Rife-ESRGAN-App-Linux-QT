from abc import ABC
from dataclasses import dataclass


@dataclass
class Encoder(ABC):
    preset_tag: str
    preInputsettings: str
    postInputSettings: str
    qualityControlMode: str = "-crf"


class copyAudio(Encoder):
    preset_tag = "copy_audio"
    preInputsettings = None
    postInputSettings = "-c:a copy"


class aac(Encoder):
    preset_tag = "aac"
    preInputsettings = None
    postInputSettings = "-c:a aac"


class libmp3lame(Encoder):
    preset_tag = "libmp3lame"
    preInputsettings = None
    postInputSettings = "-c:a libmp3lame"


class libx264(Encoder):
    preset_tag = "libx264"
    preInputsettings = None
    postInputSettings = "-c:v libx264"


class libx265(Encoder):
    preset_tag = "libx265"
    preInputsettings = None
    postInputSettings = "-c:v libx265"


class vp9(Encoder):
    preset_tag = "vp9"
    preInputsettings = None
    postInputSettings = "-c:v libvpx-vp9"
    qualityControlMode: str = "-cq:v"


class av1(Encoder):
    preset_tag = "av1"
    preInputsettings = None
    postInputSettings = "-c:v libsvtav1"


class x264_vulkan(Encoder):
    preset_tag = "x264_vulkan"
    preInputsettings = "-init_hw_device vulkan=vkdev:0 -filter_hw_device vkdev"
    postInputSettings = "-filter:v format=nv12,hwupload -c:v h264_vulkan"
    # qualityControlMode: str = "-quality" # this is not implemented very well, quality ranges from 0-4 with little difference, so quality changing is disabled.


class x264_nvenc(Encoder):
    preset_tag = "x264_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v h264_nvenc -preset slow"
    qualityControlMode: str = "-cq:v"


class x265_nvenc(Encoder):
    preset_tag = "x265_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v hevc_nvenc -preset slow"
    qualityControlMode: str = "-cq:v"


class av1_nvenc(Encoder):
    preset_tag = "av1_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v av1_nvenc -preset slow"
    qualityControlMode: str = "-cq:v"


class h264_vaapi(Encoder):
    preset_tag = "x264_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v h264_vaapi"
    qualityControlMode: str = "-qp"


class h265_vaapi(Encoder):
    preset_tag = "x265_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v hevc_vaapi"
    qualityControlMode: str = "-qp"


class av1_vaapi(Encoder):
    preset_tag = "av1_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v av1_vaapi"
    qualityControlMode: str = "-qp"


class EncoderSettings:
    def __init__(self, encoder_preset):
        self.encoder_preset = encoder_preset
        self.encoder: Encoder = self.getEncoder()

    def getEncoder(self) -> Encoder:
        for encoder in Encoder.__subclasses__():
            if encoder.preset_tag == self.encoder_preset:
                return encoder

    def getPreInputSettings(self) -> str:
        return self.encoder.preInputsettings

    def getPostInputSettings(self) -> str:
        return self.encoder.postInputSettings

    def getQualityControlMode(self) -> str:
        return self.encoder.qualityControlMode

    def getPresetTag(self) -> str:
        return self.encoder.preset_tag
