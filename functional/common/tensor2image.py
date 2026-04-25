import torch
import numpy as np

from typing import Union


def tensor2image(input_tesnor: torch.Tensor, denormlize: Union[None, str] = 'zo') -> np.ndarray:
    """
    """

    image = input_tesnor.detach().cpu().squeeze().numpy()

    image = np.transpose(image, [1,2,0])

    if denormlize == 'zo':
        image = np.clip(image*255, 0, 255)
    
    image = image.astype(np.uint8)

    return image