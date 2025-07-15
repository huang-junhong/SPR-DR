import os
import cv2
import numpy as np

from typing import List, Union


def write_image(image: np.ndarray, path: str, input_channel_order: str = "rgb") -> bool:
    """
    Writes an image to the specified file path, converting channel order if necessary.

    Inputs:
    --------
        image (np.ndarray): 
            The input image as a NumPy array.
        path (str):
            The file path where the image will be saved.
        input_channel_order (str): 
            The channel order of the input image:
            - "rgb": Input is in RGB order and will be converted to BGR for saving.
            - "bgr": Input is in BGR order (default for OpenCV), no conversion is needed.

    Returns:
    --------
        bool: 
            True if the image is successfully saved, False otherwise.
    """

    # Validate inputs
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    if input_channel_order.lower() not in ["rgb", "bgr"]:
        raise ValueError("input_channel_order must be either 'rgb' or 'bgr'.")

    try:
        # Convert RGB to BGR if necessary
        if input_channel_order.lower() == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Write the image to the specified path
        success = cv2.imwrite(path, image)
        if not success:
            print(f"Error: Failed to write the image to path: {path}")
        return success

    except Exception as e:
        print(f"Error saving image at path: {path}. Details: {e}")
        return False


def write_images(images: Union[np.ndarray, List[np.ndarray]], folder: str,
                 input_channel_order: str = "rgb", image_names: Union[None, List[str]] = None) -> List[bool]:
    """
    Writes a batch of images to the specified folder with consistent naming or provided names.

    Args:
        images (Union[np.ndarray, List[np.ndarray]]): Images to be written. Can be:
            - A 4D NumPy array of shape (N, H, W, C), where N is the number of images.
            - A list of 3D NumPy arrays, each of shape (H, W, C).
        folder (str): The folder where the images will be saved.
        input_channel_order (str): The channel order of the input images:
            - "rgb": Input is in RGB order and will be converted to BGR for saving.
            - "bgr": Input is in BGR order (default for OpenCV), no conversion is needed.
        image_names (Union[None, List[str]]): Optional list of image names (without extension).
            - If provided, its length must match the number of images.
            - If None, images will be named sequentially (e.g., 00001.jpg, 00002.jpg).

    Returns:
        list[bool]: Is successfully written each image.

    Raises:
        ValueError: If image_names is provided but its length doesn't match the number of images.
    """

    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)

    # Normalize input to a list of images
    if isinstance(images, np.ndarray) and len(images.shape) == 4:
        images = [images[i] for i in range(images.shape[0])]

    # Validate image_names length if provided
    if image_names is not None and len(image_names) != len(images):
        raise ValueError("The length of image_names must match the number of images.")

    # Generate sequential names if image_names is not provided
    if image_names is None:
        num_images = len(images)
        num_digits = len(str(num_images))  # Determine the number of digits for zero-padding
        image_names = [str(i + 1).zfill(num_digits) for i in range(num_images)]

    success = []  # To track if all images are successfully saved

    for i, (image, name) in enumerate(zip(images, image_names)):
        try:
            # Construct the full file path
            file_path = os.path.join(folder, f"{name}.jpg")

            write_image(image, file_path, input_channel_order)

            success.append(True)

        except Exception as e:
            print(f"Error saving image {name}. Details: {e}")
            success.append(False)

    return success

