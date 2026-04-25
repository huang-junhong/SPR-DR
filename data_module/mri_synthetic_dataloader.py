import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import random
import copy
import tqdm
import threading

from PIL import Image
from typing import List
from torch.utils import data as data


class MRISyntheticDataset(data.Dataset):
    def __init__(self, data_path: str, isRot:bool=True, isFlip:bool=True):

        self.isRot = isRot
        self.isFlip = isFlip

        print('Prepare Trainset')
        _LR2_PATH = []
        _LR4_PATH = []
        _HR_PATH = []
        for i in range(len(data_path)):
            print(f"Prepare: {data_path[i]}")
            _LR2_PATH.extend(self._load_img_path(data_path[i]+'/LR2_PATH.txt'))
            _LR4_PATH.extend(self._load_img_path(data_path[i]+'/LR4_PATH.txt'))
            _HR_PATH.extend(self._load_img_path(data_path[i]+'/HR_PATH.txt'))
        self.num_of_err = 0
        print('The num of LR2s/LR4s/HRs:{}/{}/{}'.format(len(_LR2_PATH),len(_LR4_PATH),len(_HR_PATH)))
        assert len(_LR2_PATH) == len(_LR4_PATH) == len(_HR_PATH)
        self.LR2_PATH = np.array(_LR2_PATH)
        self.LR4_PATH = np.array(_LR4_PATH)
        self.HR_PATH  = np.array(_HR_PATH)
        print("Synthetic Dataset prepare ok")
        print("---"*30)
    
    #-------------------------------------------------------------------------
    def __len__(self):
        return len(self.LR2_PATH)
    
    def __getitem__(self, index):
        lr2 = cv2.imread(self.LR2_PATH[index])
        lr4 = cv2.imread(self.LR4_PATH[index])
        hr  = cv2.imread(self.HR_PATH[index])

        #---------------
        # Error Check
        #---------------
        while(lr2 is None or lr4 is None or hr is None):
            # TODO: ERROR LOG

            #-------------
            # Reload image
            #-------------
            new_index = random.randint(0, self.__len__())
            lr2 = cv2.imread(self.LR2_PATH[new_index])
            lr4 = cv2.imread(self.LR4_PATH[new_index])
            hr  = cv2.imread(self.HR_PATH[new_index])

        lr2 = cv2.cvtColor(lr2, cv2.COLOR_BGR2RGB)
        lr4 = cv2.cvtColor(lr4, cv2.COLOR_BGR2RGB)
        hr  = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        if self.isRot:
            rot = random.randint(0, 3)
            lr2 = np.rot90(lr2, rot, [0,1])
            lr4 = np.rot90(lr4, rot, [0,1])
            hr  = np.rot90(hr, rot, [0,1])
            
        if self.isFlip:
            filp = random.randint(-1, 2)
            if filp != 2:
                lr2 = cv2.flip(lr2, filp)
                lr4 = cv2.flip(lr4, filp)
                hr  = cv2.flip(hr, filp)

        lr2 = np.transpose(lr2, [2,0,1]).astype('float32')
        lr4 = np.transpose(lr4, [2,0,1]).astype('float32')
        hr  = np.transpose(hr, [2,0,1]).astype('float32')
        
        lr2 = lr2 / 255.
        lr4 = lr4 / 255.
        hr  = hr / 255.
        
        return lr2, lr4, hr

    #-------------------------------------------------------------------------
    def _load_img_path(self, path) -> List[str]:
        files = []
        with open(path, 'r') as f:
            for line in f.readlines():
                files.append(line.strip('\n'))
        return files
    

class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.loader_iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def restart(self):
        self.loader_iter = iter(self.loader)

    def preload(self):
        try:
            self.next_lr2, self.next_lr4, self.next_hr = next(self.loader_iter)
        except:
            self.next_lr2 = None
            self.next_lr4 = None
            self.next_hr  = None
            return
        with torch.cuda.stream(self.stream):
            self.next_lr2 = self.next_lr2.cuda(non_blocking=True)
            self.next_lr4 = self.next_lr4.cuda(non_blocking=True)
            self.next_hr  = self.next_hr.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        lr2 = self.next_lr2
        lr4 = self.next_lr4
        hr = self.next_hr
        self.preload()
        if lr2 is None or lr4 is None:
            return None, None, None
        return lr2.detach(), lr4.detach(), hr.detach()      