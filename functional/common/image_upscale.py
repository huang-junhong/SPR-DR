import torch
import torch.nn.functional as F

def image_upsample_torch(lr: torch.Tensor, srf: int, up_mode: str='bicubic'):
    """
    """

    return F.interpolate(lr, scale_factor=srf, mode=up_mode, align_corners=False if up_mode in ('bilinear','bicubic') else None)
