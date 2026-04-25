import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import file_io

from torch.utils import data as data
from typing import Tuple, Union, Dict

from basicsr.utils import img2tensor
from basicsr.data.transforms import augment
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt


class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.
    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.
    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, text_path, folder_path=None):
        super(RealESRGANDataset, self).__init__() 
        # Load HR path
        if text_path is not None:
            if isinstance(text_path, list):
                self.HR_PATH = []
                for path in text_path:
                    self.HR_PATH.extend(self._load_img_path(path=os.path.join(path, 'HR_PATH.txt')))
            else:
                self.HR_PATH = self._load_img_path(text_path)
        if folder_path is not None:
            self.HR_PATH = file_io.base.get_file_path_by_suffix()

        # blur settings for the first degradation
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  # a list for each kernel probability
        self.blur_sigma = [0.2, 3]
        self.resize_prob = [0.2, 0.7, 0.1]
        self.resize_range = [0.15, 1.5]
        self.betag_range = [0.5, 4]  # betag used in generalized Gaussian blur kernels
        self.betap_range = [1, 2]  # betap used in plateau blur kernels
        self.sinc_prob = 0.1  # the probability for sinc filters
        self.gray_noise_prob = 0.4
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.jpeg_range = [30, 95]

        # blur settings for the second degradation
        self.second_blur_prob = 0.8
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1
        self.gray_noise_prob2 = 0.4
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.jpeg_range2 = [30, 95]

        # a final sinc filter
        self.final_sinc_prob = 0.8

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_gt = cv2.imread(self.HR_PATH[index])
        while img_gt is None:
            new_index = random.randint(0,self.__len__())
            img_gt = cv2.imread(self.HR_PATH[new_index])

        img_gt = img_gt.astype('float32') / 255.

        # -------------------- Do augmentation for training: flip, rotation -------------------- #

        img_gt = augment(img_gt, True, False)

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}
        return return_d

    def _load_img_path(self, path):
        files = []
        with open(path, 'r') as f:
            for line in f.readlines():
                files.append(line.strip('\n'))
        return files
    
    def __len__(self):
        return len(self.HR_PATH)
    

class Degeration(nn.Module):
    def __init__(self, hr_size=256, srf=4, usm=False):
        super(Degeration, self).__init__()
        self.hr_size = hr_size
        self.srf = srf
        self.usm = usm

        self.USM = USMSharp().cuda()
        self.jpeger = DiffJPEG(differentiable=False).cuda()

        # blur settings for the first degradation
        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  # a list for each kernel probability
        self.blur_sigma = [0.2, 3]
        self.resize_prob = [0.2, 0.7, 0.1]
        self.resize_range = [0.15, 1.5]
        self.betag_range = [0.5, 4]  # betag used in generalized Gaussian blur kernels
        self.betap_range = [1, 2]  # betap used in plateau blur kernels
        self.sinc_prob = 0.1  # the probability for sinc filters
        self.gray_noise_prob = 0.4
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.jpeg_range = [30, 95]

        # blur settings for the second degradation
        self.second_blur_prob = 0.8
        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1
        self.gray_noise_prob2 = 0.4
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.jpeg_range2 = [30, 95]

        # a final sinc filter
        self.final_sinc_prob = 0.8

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def forward(self, deg_info):
        hr = deg_info['gt'].cuda()
        hr_usm = self.USM(hr)

        self.kernel_1 = deg_info['kernel1'].cuda()
        self.kernel_2 = deg_info['kernel2'].cuda()
        self.sinc_kernel = deg_info['sinc_kernel'].cuda()

        ori_h, ori_w = hr.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
        # blur
        if self.usm:
            current_lr = filter2D(hr_usm, self.kernel_1)
        else:
            current_lr = filter2D(hr, self.kernel_1)

        # random resize
        current_lr = self._resize_img(current_lr, self.resize_prob, self.resize_range)
        
        # noise
        current_lr = self._add_noise(current_lr, self.gray_noise_prob, self.gaussian_noise_prob, self.noise_range, self.poisson_scale_range)

        # JPGE compression
        current_lr = self._jpge_compress(current_lr, self.jpeg_range)

    # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            current_lr = filter2D(current_lr, self.kernel_2)
        
        # resize
        current_lr = self._resize_img(current_lr, self.resize_prob2, self.resize_range2)

        # noise
        current_lr = self._add_noise(current_lr, self.gray_noise_prob2, self.gaussian_noise_prob2, self.noise_range2, self.poisson_scale_range2)

        # final sinc
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            current_lr = F.interpolate(current_lr, size=(ori_h // self.srf, ori_w // self.srf), mode=mode)
            current_lr = filter2D(current_lr, self.sinc_kernel)
            # JPEG compression
            current_lr = self._jpge_compress(current_lr, self.jpeg_range2)

        else:
            # JPEG compression
            current_lr = self._jpge_compress(current_lr, self.jpeg_range2)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            current_lr = F.interpolate(current_lr, size=(ori_h // self.srf, ori_w // self.srf), mode=mode)
            current_lr = filter2D(current_lr, self.sinc_kernel)

        # get final data
        current_lr = torch.clamp((current_lr * 255.0).round(), 0, 255) / 255.
        (hr, hr_usm), lr = paired_random_crop([hr, hr_usm], current_lr, self.hr_size, self.srf)
        return lr.detach(), hr_usm.detach(), hr.detach()


    def _resize_img(self, img, prob, range):
        updown_type = random.choices(['up', 'down', 'keep'], prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(range[0], 1)
        else:
            scale = 1

        mode = random.choice(['area', 'bilinear', 'bicubic'])
        img = F.interpolate(img, scale_factor=scale, mode=mode)

        return img

    def _add_noise(self, img, gray_prob, gaussian_prob, gaussian_range, poisson_range):

        gray_noise_prob = gray_prob

        if np.random.uniform() < gaussian_prob:
            img = random_add_gaussian_noise_pt(
                  img,
                  sigma_range=gaussian_range,
                  clip=True,
                  rounds=False,
                  gray_prob=gray_noise_prob)
        else:
            img = random_add_poisson_noise_pt(
                  img,
                  scale_range=poisson_range,
                  gray_prob=gray_noise_prob,
                  clip=True,
                  rounds=False)
        return img

    def _jpge_compress(self, img, jpeg_range):
        jpeg_p = img.new_zeros(img.size(0)).uniform_(*jpeg_range)
        img = torch.clamp(img, 0, 1)
        img = self.jpeger(img, quality=jpeg_p)
        return img


class data_prefetcher():
    def __init__(self, loader, hr_size=256, srf=4, usm=False):
        self.loader = loader
        self.loader_iter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.degeration = Degeration(hr_size, srf, usm)
        self.preload()

    def restart(self):
        self.loader_iter = iter(self.loader)

    def preload(self):
        try:
            deg_info = next(self.loader_iter)
            self.next_lr, self.next_hr_usm, self.next_hr = self.degeration(deg_info)
        except StopIteration:
            self.next_lr = None
            self.next_hr = None
            self.next_hr_usm = None
            return
        with torch.cuda.stream(self.stream):
            self.next_lr = self.next_lr.cuda(non_blocking=True)
            self.next_hr = self.next_hr.cuda(non_blocking=True)
            self.next_hr_usm = self.next_hr_usm.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        lr = self.next_lr
        hr = self.next_hr
        hr_usm = self.next_hr_usm
        self.preload()
        
        if lr == None:
            return None, None, None
        
        return lr.detach(), hr_usm.detach(), hr.detach()  
    
    