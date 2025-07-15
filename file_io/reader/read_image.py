import cv2
import numpy as np

from typing import List, Union

from file_io.common import get_file_path_by_suffix


def read_image(path: str, as_rgb: bool = True, as_gray: bool = False, 
               read_as_unchange: bool = False) -> np.ndarray:
    """
    Reads an image from the specified file path with options for RGB, grayscale, or unchanged modes.

    Inputs:
    --------
        path (str): The file path of the image to read.
        as_rgb (bool): If True, reads the image in RGB mode (default: True). Ignored if `as_gray` or `read_as_unchange` is True.
        as_gray (bool): If True, reads the image in grayscale mode. Overrides `as_rgb`.
        read_as_unchange (bool): If True, reads the image in the unchanged mode (preserving alpha channel if present).
                                  Overrides both `as_rgb` and `as_gray`.

    Returns:
    --------
        np.ndarray: The loaded image as a NumPy array. Returns None if the image cannot be read.
    """

    # Determine the appropriate flag for cv2.imread based on the input arguments
    if read_as_unchange:
        flag = cv2.IMREAD_UNCHANGED  
    elif as_gray:
        flag = cv2.IMREAD_GRAYSCALE
    else:
        flag = cv2.IMREAD_COLOR

    image = cv2.imread(path, flag)

    # Check if the image was successfully read
    if image is None:
        print(f"Warning: Unable to read image at path: {path}")
        return None
    
    if as_rgb and not as_gray and not read_as_unchange:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"Error converting image to RGB at path: {path}. Error: {e}")
            return None

    return image


def read_images(folder: str, suffix: Union[str, List[str]], as_rgb: bool = True, 
                as_gray: bool = False, read_as_unchange: bool = False) -> List[np.ndarray]:
    """
    Reads all images in a folder that match the specified suffixes with flexible reading modes.

    Args:
    --------
        folder (str): Path to the folder containing images.
        suffix (Union[str, List[str]]): A single file suffix (e.g., '.png') or a list of suffixes (e.g., ['.png', '.jpg']).
        as_rgb (bool): If True, reads images in RGB mode. Ignored if `as_gray` or `read_as_unchange` is True.
        as_gray (bool): If True, reads images in grayscale mode. Overrides `as_rgb`.
        read_as_unchange (bool): If True, reads images in the unchanged mode (preserving alpha channel if present).
                                  Overrides both `as_rgb` and `as_gray`.

    Returns:
    --------
        List[np.ndarray]: A list of loaded images as NumPy arrays. Excludes images that could not be read.
    """
    
    # Retrieve file paths for all images with the specified suffix
    image_paths = get_file_path_by_suffix(folder, suffix)
    images = []  # List to store loaded images

    for path in image_paths:
        # Read each image with the specified mode
        image = read_image(path, as_rgb=as_rgb, as_gray=as_gray, read_as_unchange=read_as_unchange)
        if image is not None:  # Only include successfully read images
            images.append(image)

    return images

