import copy
import torch
import numpy as np

from typing import List, Union


def image_handle(images: Union[np.ndarray, List[np.ndarray]], normlize: str = 'zo', 
                 chw: bool = True, expand_dim: Union[None, int] = None, to_tensor: bool = True):
    """
    """

    if isinstance(images, np.ndarray):
        img = copy.deepcopy(images)
        img = img.astype(np.float32)

        # TODO: write a common function
        if normlize == 'zo':
            img /= 255.

        if chw:
            img = np.transpose(img, [2,0,1])

        if expand_dim is not None:
            img = np.expand_dims(img, axis=expand_dim)
        
        if to_tensor:
            img = torch.from_numpy(img)

        return img

    elif isinstance(images, List):
        results = []
        for img in images:
            results.append(image_handle(img, normlize, chw, expand_dim, to_tensor))
        return results
