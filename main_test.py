import os
import cv2
import torch
import pyiqa
import argparse
import numpy as np

from typing import Union, List, Optional, Any

import file_io
import torch_model


SUPPORT_IMAGE_FORMAT = ['.png', '.jpg', '.jpeg', '.bmp']


def image2tensor(images: Union[np.ndarray, List[np.ndarray]], normlize: Optional[str] = None, dtype: str = "f32") -> torch.Tensor:
    """
    Converts an image or a list of images from NumPy array(s) to a PyTorch tensor with optional normalization.
    
    Inputs
    -----------
    images : np.ndarray or list of np.ndarray
        Input image as a NumPy array (shape: (H, W, C) or (H, W)) or a list of such arrays.
    normlize : str, optional
        Normalization mode:
            - "A": Normalize image to [0, 1].
            - "B": Normalize image to [-1, 1].
            - None: No normalization (default).
    dtype : str, optional
        Desired data type for the tensor. Options:
            - "f16": Converts to float16.
            - "bf16": Converts to bfloat16.
            - "f32": Converts to float32 (default).
            - "f64": Converts to float64.
            - "i32": Converts to int32.
            - "i64": Converts to int64.
    
    Returns
    --------
    torch.Tensor
        For a single image, returns a tensor (with shape (C, H, W) if the image is 3D).
        For a list of images, returns a tensor with images stacked along the first dimension (N,C,H,W).
    
    Raises
    --------
    ValueError
        If the input is not a NumPy array or a list of NumPy arrays,
        or if the normalization mode or dtype is invalid.
    """

    # Define valid data types mapping
    valid_dtypes = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "f64": torch.float64,
        "i32": torch.int32,
        "i64": torch.int64
    }
    if dtype not in valid_dtypes:
        raise ValueError(f"Invalid dtype '{dtype}'. Must be one of: {list(valid_dtypes.keys())}")
    
    # Validate normalization mode
    if normlize not in [None, "A", "B"]:
        raise ValueError("Normalization mode must be None, 'A', or 'B'.")
    
    def process_single_image(image: np.ndarray) -> torch.Tensor:
        # Validate that the input is a NumPy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a NumPy array.")
        
        # Convert the NumPy array to a PyTorch tensor and cast it to the desired dtype
        tensor = torch.from_numpy(image).to(dtype=valid_dtypes[dtype])
        
        # Apply normalization if specified
        if normlize == "A":
            tensor = tensor / 255.0  # Normalize to [0, 1]
        elif normlize == "B":
            tensor = tensor / 127.5 - 1.0  # Normalize to [-1, 1]
        
        # If the image has three dimensions (assumed to be H, W, C), permute to (C, H, W)
        if len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1)
        elif len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0)

        return tensor

    # Process the input based on its type
    if isinstance(images, np.ndarray):
        return process_single_image(images)
    elif isinstance(images, list):
        tensors = [process_single_image(img).unsqueeze(0) for img in images]
        return torch.cat(tensors, dim=0)
    else:
        raise ValueError("Input must be a NumPy array or a list of NumPy arrays.")


def tensor2image(input: torch.Tensor, denormlize: str = "A") -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array representing an image.

    Inputs
    --------
        input (torch.Tensor): Input tensor to be converted. Expected shape:
                              - (C, H, W) for single image
                              - (N, C, H, W) for batch of images
        denormlize (str): Denormalization method to apply:
                          - None: No denormalization
                          - "A": Assumes input is in the range [0, 1], scales to [0, 255]
                          - "B": Assumes input is in the range [-1, 1], scales to [0, 255]

    Returns
    --------
        np.ndarray: 
            Converted NumPy array representing the image(s).
            - Shape (H, W, C) for single image
            - Shape (N, H, W, C) for batch of images

    Raises:
        ValueError: If the input is not a torch.Tensor or denormlize is invalid.
    """
    # Ensure input is a PyTorch tensor
    if not isinstance(input, torch.Tensor):
        raise ValueError(f"Input's type should be torch.Tensor, but {type(input)}")

    # Clone the tensor to avoid modifying the original and convert to float32
    array = input.clone().detach().to(torch.float32)

    # Apply denormalization
    if denormlize is None:
        pass  # No change
    elif denormlize == 'A':  # Scale [0, 1] to [0, 255]
        array = array * 255.0
    elif denormlize == 'B':  # Scale [-1, 1] to [0, 255]
        array = (array + 1.0) * 127.5
    else:
        raise ValueError(f"denormlize should be one of [None, 'A', 'B'], but is {denormlize}")

    # Convert tensor to NumPy array
    array: np.ndarray = array.cpu().numpy()

    # Clip values to ensure they fall within [0, 255] and convert to uint8
    array = np.clip(array, 0, 255).astype(np.uint8)

    # Adjust dimensions for single image or batch of images
    if len(array.shape) == 3:
        # Single image (C, H, W) -> (H, W, C)
        array = np.transpose(array, (1, 2, 0))
    elif len(array.shape) == 4:
        # Batch of images (N, C, H, W) -> (N, H, W, C)
        array = np.transpose(array, (0, 2, 3, 1))

    return array


def prepare_data(lr_folder: str, hr_folder: Union[None, str]):
    """
    """

    lr_paths = file_io.common.get_file_path_by_suffix(folder=lr_folder, suffix=SUPPORT_IMAGE_FORMAT)

    if hr_folder is not None:
        hr_paths = file_io.common.get_file_path_by_suffix(folder=hr_folder, suffix=SUPPORT_IMAGE_FORMAT)
        assert len(lr_paths) == len(hr_paths), f"the number of image in hr folder and lr folder must same"
    else:
        hr_paths = None

    return lr_paths, hr_paths


def prepare_model(model_type: str, model_path: str, srf: int):
    if model_type.lower() == 'srres':
        if srf == 2:
            model = torch_model.SRRes_SRF2()
        elif srf == 4:
            model = torch_model.SRRes()
        else:
            raise ValueError("The SRF unsupport")
    elif model_type.lower() == 'rrdn':
        if srf == 2:
            model = torch_model.RRDBNet_SRF2(3, 3, 64, 23)
        elif srf == 4:
            model = torch_model.RRDBNet(3, 3, 64, 23)
        else:
            raise ValueError("The SRF unsupport")
    elif model_type.lower() == 'swinirs':
        if srf == 2:
            model = torch_model.SwinIR(
                upscale=2, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], 
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv'
            )
        elif srf == 4:
            model = torch_model.SwinIR(
                upscale=4, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], 
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv'
            )
        else:
            raise ValueError("The SRF unsupport")
    elif model_type.lower() == 'swinirm':
        if srf == 2:
            model = torch_model.SwinIR(
                upscale=2, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], 
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
            )
        elif srf == 4:
            model = torch_model.SwinIR(
                upscale=4, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],  
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
            )
        else:
            raise ValueError("The SRF unsupport")
    else:
        raise ValueError('model type unsupport')

    model.load_state_dict(torch.load(model_path))

    return model


@torch.no_grad()
def test_process(lr_paths: List[str], model: Any, save_folder: str):

    os.makedirs(save_folder, exist_ok=True)
    img_num = len(lr_paths)

    for idx, lp in enumerate(lr_paths):
        print(f"[{idx+1}/{img_num}] processing {os.path.basename(lp)}")
        lr = image2tensor(file_io.reader.read_image(lp), normlize='A', dtype='f32').unsqueeze(0)
        sr = tensor2image(model(lr)).squeeze(0)

        cv2.imwrite(os.path.join(save_folder, os.path.basename(lp)), img=cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))


def calculate_iqas(sr_folder: str, hr_folder: str, iqas: List[str]):
    """
    """

    for iqa in iqas:
        print(f"Calculate {iqa}...")
        if iqa.lower() == 'psnr':
            cal_iqa = pyiqa.create_metric(metric_name=iqa, as_loss=False, device=torch.device('cuda'), color_space='ycbcr', test_y_channel=True)
        else:
            cal_iqa = pyiqa.create_metric(metric_name=iqa, as_loss=False, device=torch.device('cuda'))

        values = []

        srs = file_io.common.get_file_path_by_suffix(folder=sr_folder, suffix=SUPPORT_IMAGE_FORMAT)
        if hr_folder is not None:
            hrs = file_io.common.get_file_path_by_suffix(folder=hr_folder, suffix=SUPPORT_IMAGE_FORMAT)
        else:
            hrs = [None] * len(srs)

        for sr_path, hr_path in zip(srs, hrs):

            if hr_path is not None:
                v = cal_iqa(sr_path, hr_path)
            else:
                v = cal_iqa(sr_path)
            values.append(float(v))

        print(f"{iqa} mean value is: {sum(values)/len(values)}")
        file_io.writer.write_excel(path=os.path.join(f"{sr_folder}", "IQAs.xlsx"),
                                   sheet_name=iqa,
                                   data=[[v] for v in values])

        
    def model_test(lr_folder: str, hr_folder: str, save_folder: str,
                model_type: str, model_path: str, srf: int,
                iqas: List[str]):
        
        lr_paths, hr_paths = prepare_data(lr_folder=lr_folder, hr_folder=hr_folder)
        model = prepare_model(model_type=model_type, model_path=model_path, srf=srf)
        test_process(lr_paths, model, save_folder)
        calculate_iqas(save_folder, hr_folder, iqas)

        print("complete")


    def parse_args():
        parser = argparse.ArgumentParser(
            description="Test trained models"
        )

        parser.add_argument("-mp", "--model_path", 
                            help="The path of test model")
        
        parser.add_argument("-mt", "--model_type", 
                            help="The model structure, it must in ['SRRes', 'RRDN', SwinIRS, SwinIRM] ")
        
        parser.add_argument("-srf", "--SRF", type=int,
                            help="The super-resolution factor of model")
        
        parser.add_argument("-lf", "--lr_folder", 
                            help="The folder of low-resolution images")
        
        parser.add_argument("-sf", "--save_folder",
                            help="The path to save super-resolution images")
        
        parser.add_argument("-hf", "--hr_folder",
                            help="The path of correspond high-resolution images")
        
        parser.add_argument("-iqa", "--IQAs", nargs="+", default=['psnr', 'ssim', 'lpips', 'dists'],
                            help="The IQAs to evaluate. Use space to separate multiple values.")

        return parser.parse_args()


    if __name__ == "__main__":
        args = parse_args()

        model_test(lr_folder=args.lr_folder, hr_folder=args.hr_folder, save_folder=args.save_folder,
                model_type=args.model_type, model_path=args.model_path, srf=args.srf, iqas=args.IQAs)
