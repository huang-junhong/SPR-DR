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


class SyntheticDataset(data.Dataset):
    def __init__(self, data_path: str, isRot:bool=True, isFlip:bool=True):

        self.isRot = isRot
        self.isFlip = isFlip

        print('Prepare Trainset')
        _LR_PATH = []
        _HR_PATH = []
        for i in range(len(data_path)):
            print(f"Prepare: {data_path[i]}")
            _LR_PATH.extend(self._load_img_path(data_path[i]+'/LR_PATH.txt'))
            _HR_PATH.extend(self._load_img_path(data_path[i]+'/HR_PATH.txt'))
        self.num_of_err = 0
        print('The num of LRs/HRs:{}/{}'.format(len(_LR_PATH),len(_HR_PATH)))
        assert len(_LR_PATH) == len(_HR_PATH)
        self.LR_PATH = np.array(_LR_PATH)
        self.HR_PATH = np.array(_HR_PATH)
        print("Synthetic Dataset prepare ok")
        print("---"*30)
    
    #-------------------------------------------------------------------------
    def __len__(self):
        return len(self.LR_PATH)
    
    def __getitem__(self, index):
        lr = cv2.imread(self.LR_PATH[index])
        hr = cv2.imread(self.HR_PATH[index])

        #---------------
        # Error Check
        #---------------
        while(lr is None or hr is None):
            # TODO: ERROR LOG

            #-------------
            # Reload image
            #-------------
            new_index = random.randint(0, self.__len__())
            lr = cv2.imread(self.LR_PATH[new_index])
            hr = cv2.imread(self.HR_PATH[new_index])

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        if self.isRot:
            rot = random.randint(0, 3)
            lr = np.rot90(lr, rot, [0,1])
            hr = np.rot90(hr, rot, [0,1])
            
        if self.isFlip:
            filp = random.randint(-1, 2)
            if filp != 2:
                lr = cv2.flip(lr, filp)
                hr = cv2.flip(hr, filp)

        lr = np.transpose(lr, [2,0,1]).astype('float32')
        hr = np.transpose(hr, [2,0,1]).astype('float32')
        
        lr = lr / 255.
        hr = hr / 255.
        
        return lr, hr

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
            self.next_lr, self.next_hr = next(self.loader_iter)
        except:
            self.next_lr = None
            self.next_hr = None
            return
        with torch.cuda.stream(self.stream):
            self.next_lr = self.next_lr.cuda(non_blocking=True)
            self.next_hr = self.next_hr.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        lr = self.next_lr
        hr = self.next_hr
        self.preload()
        if lr is None:
            return None, None
        return lr.detach(), hr.detach()      