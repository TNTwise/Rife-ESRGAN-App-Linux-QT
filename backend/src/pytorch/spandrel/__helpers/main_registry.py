from __future__ import annotations


from ..architectures import (
    ATD,
    CRAFT,
    DAT,
    DCTLSA,
    DITN,
    DRCT,
    ESRGAN,
    FBCNN,
    GFPGAN,
    GRL,
    HAT,
    IPT,
    PLKSR,
    RGT,
    SAFMN,
    SAFMNBCIE,
    SPAN,
    SPANPlus,
    sudo_SPANPlus,
    Compact,
    DnCNN,
    DRUNet,
    FFTformer,
    HVICIDNet,
    KBNet,
    LaMa,
    MixDehazeNet,
    MMRealSR,
    MoSR,
    NAFNet,
    OmniSR,
    RealCUGAN,
    RestoreFormer,
    RetinexFormer,
    SCUNet,
    SeemoRe,
    SwiftSRGAN,
    Swin2SR,
    SwinIR,
    Uformer,
    Sebica,
    RTMoSR,
    RCAN,
)
from .registry import ArchRegistry, ArchSupport

MAIN_REGISTRY = ArchRegistry()
"""
The main architecture registry of spandrel.

Modifying this registry will affect all `ModelLoader` instances without a custom registry.
"""

MAIN_REGISTRY.add(
    ArchSupport.from_architecture(Compact.CompactArch()),
    ArchSupport.from_architecture(SwiftSRGAN.SwiftSRGANArch()),
    ArchSupport.from_architecture(HAT.HATArch()),
    ArchSupport.from_architecture(GRL.GRLArch()),
    ArchSupport.from_architecture(Swin2SR.Swin2SRArch()),
    ArchSupport.from_architecture(SwinIR.SwinIRArch()),
    ArchSupport.from_architecture(GFPGAN.GFPGANArch()),
    ArchSupport.from_architecture(RestoreFormer.RestoreFormerArch()),
    ArchSupport.from_architecture(LaMa.LaMaArch()),
    ArchSupport.from_architecture(OmniSR.OmniSRArch()),
    ArchSupport.from_architecture(SCUNet.SCUNetArch()),
    ArchSupport.from_architecture(FBCNN.FBCNNArch()),
    ArchSupport.from_architecture(Uformer.UformerArch()),
    ArchSupport.from_architecture(RGT.RGTArch()),
    ArchSupport.from_architecture(DAT.DATArch()),
    ArchSupport.from_architecture(CRAFT.CRAFTArch()),
    ArchSupport.from_architecture(KBNet.KBNetArch()),
    ArchSupport.from_architecture(DITN.DITNArch()),
    ArchSupport.from_architecture(MMRealSR.MMRealSRArch()),
    ArchSupport.from_architecture(SPAN.SPANArch()),
    ArchSupport.from_architecture(RealCUGAN.RealCUGANArch()),
    ArchSupport.from_architecture(SAFMN.SAFMNArch()),
    ArchSupport.from_architecture(SAFMNBCIE.SAFMNBCIEArch()),
    ArchSupport.from_architecture(DCTLSA.DCTLSAArch()),
    ArchSupport.from_architecture(FFTformer.FFTformerArch()),
    ArchSupport.from_architecture(NAFNet.NAFNetArch()),
    ArchSupport.from_architecture(ATD.ATDArch()),
    ArchSupport.from_architecture(MixDehazeNet.MixDehazeNetArch()),
    ArchSupport.from_architecture(DRUNet.DRUNetArch()),
    ArchSupport.from_architecture(DnCNN.DnCNNArch()),
    ArchSupport.from_architecture(IPT.IPTArch()),
    ArchSupport.from_architecture(DRCT.DRCTArch()),
    ArchSupport.from_architecture(ESRGAN.ESRGANArch()),
    ArchSupport.from_architecture(PLKSR.PLKSRArch()),
    ArchSupport.from_architecture(RetinexFormer.RetinexFormerArch()),
    ArchSupport.from_architecture(HVICIDNet.HVICIDNetArch()),
    ArchSupport.from_architecture(SeemoRe.SeemoReArch()),
    ArchSupport.from_architecture(MoSR.MoSRArch()),
    ArchSupport.from_architecture(sudo_SPANPlus.sudo_SPANPlusArch()),
    ArchSupport.from_architecture(SPANPlus.SPANPlusArch()),
    ArchSupport.from_architecture(Sebica.SebicaArch()),
    ArchSupport.from_architecture(RTMoSR.RTMoSRArch()),
    ArchSupport.from_architecture(RCAN.RCANArch()),
)
