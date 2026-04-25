import torch
import numpy as np

from typing import Union


def crop_two_image_to_same_resolution(image_1: Union[np.ndarray, torch.Tensor],
                                      image_2: Union[np.ndarray, torch.Tensor],
                                      max_diff_limit:int=24):
    """
    """

    if isinstance(image_1, np.ndarray):
        h1, w1 = image_1.shape[-2:]
    elif isinstance(image_1, torch.Tensor):
        h1, w1 = image_1.size()[-2:]

    if isinstance(image_2, np.ndarray):
        h2, w2 = image_2.shape[-2:]
    elif isinstance(image_2, torch.Tensor):
        h2, w2 = image_2.size()[-2:]

    if abs(h1 - h2) > max_diff_limit or abs(w1 - w2) > max_diff_limit:
        raise ValueError(f"Image resolution difference exceeds max_diff_limit of {max_diff_limit}")
    
    target_h = min(h1, h2)
    target_w = min(w1, w2)

    croped_image_1 = image_1[..., :target_h, :target_w]
    croped_image_2 = image_2[..., :target_h, :target_w]

    return croped_image_1, croped_image_2
    