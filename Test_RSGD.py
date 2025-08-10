import cv2
import torch
import random
import numpy as np

import RRDB_NET
import tensor_unit
import File_function as ff

from tqdm import tqdm

from torch_model import Stander_Discriminator_with_dynamic_dropout


def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, [2,0,1])
    img = img.astype('float32') / 255.
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    return img

def crop_img_pair(lr, hr, size):
    _, _, h, w = lr.size()
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    return lr[:,:,y:y+size,x:x+size].detach(), hr[:,:,y*4:y*4+size*4,x*4:x*4+size*4].detach()



def load_imgs(lr_path:list[str], hr_path:list[str], num:int, patch_size:int):
    assert len(lr_path) == len(hr_path)

    lrs = []
    hrs = []

    for i in range(num):
        index = random.randint(0, len(lr_path)-1)

        lr = load_img(lr_path[index])
        hr = load_img(hr_path[index])

        lr, hr = crop_img_pair(lr, hr, patch_size)

        lrs.append(lr)
        hrs.append(hr)
    
    lrs = torch.cat(lrs, 0)
    hrs = torch.cat(hrs, 0)
    return lrs, hrs


def get_kern_reverse(input: dict, kern_info: dict, index: int):
    
    new_kern = {}

    for key in kern_info.keys():
        if key == 0:
            kern = set(range(kern_info[key]))
        else:
            kern = set(range(kern_info[key][index]))

        kern = kern - set(input[str(key)])
        new_kern[str(key)] = list(kern)

    return new_kern


def analyse_result(judge1, judge2):

    relationship = torch.sigmoid(judge1 - judge2).detach().cpu().squeeze().numpy()

    connection = np.where(relationship > 0.5, 1, 0).flatten()

    c_portion = np.sum(connection) / connection.shape[0]

    return c_portion


def analyse_result_2(judge1, judge2, judge3, judge4):

    relationship1 = judge1 - judge2
    relationship2 = judge3 - judge4

    relationship = torch.sigmoid(relationship1 - relationship2).detach().cpu().squeeze().numpy()

    connection = np.where(relationship > 0.5, 1, 0).flatten()

    c_portion = np.sum(connection) / connection.shape[0]

    return c_portion


def RSGD_Test():


    ##########################################
    ## load model & data
    ##########################################

    G = RRDB_NET.RRDBNet(3, 3, 64, 23)
    G.load_state_dict(torch.load(r"\\192.168.31.97\sata11-156XXXX1083\SISR\SPR-DR\307\G.pth"))
    G = G.cuda()

    D = Stander_Discriminator_with_dynamic_dropout()
    D.load_state_dict((torch.load(r"\\192.168.31.97\sata11-156XXXX1083\SISR\SPR-DR\307\Dvan.pth")))
    D = D.cuda()

    lrs = r"E:\Super-Resolution\Datasets\Test_Set\BSD100\BSD100_X4_LR\RGB"
    hrs = r"E:\Super-Resolution\Datasets\Test_Set\BSD100\BSD100_HR\RGB"

    lrs_path = ff.load_file_path(lrs)
    hrs_path = ff.load_file_path(hrs)

    ##########################################
    ## init parameter
    ##########################################

    replace_16 = tensor_unit.Replacement_v2(r_size=(16,16), size=(128,128), roll_prob=0)
    replace_4  = tensor_unit.Replacement_v2(r_size=(4,4), size=(128,128), roll_prob=0)

    num_dict = {0: 64,
                1: [64, 128],
                2: [128, 256],
                3: [256, 512],
                4: [512, 512],
                5: [512, 512]}
    
    block_portion = 0.5
    runtimes = 100

    rb_portion = []
    rb_portion_all = []

    hr_sr_relative = []
    sr_ins16_relative = []
    ins16_ins4_relative = []


    for i in tqdm(range(runtimes)):
        lrs, hrs = load_imgs(lrs_path, hrs_path, 16, 32)
        lrs, hrs = lrs.cuda(), hrs.cuda()

        srs = G(lrs).detach()

        ins16 = replace_16(hrs, srs, real_portion="adaptive", mode="spr", Dvan=D).detach()
        ins4  = replace_4(hrs, srs, real_portion="adaptive", mode="spr", Dvan=D).detach()

        hrs.requires_grad = True
        srs.requires_grad = True
        ins16.requires_grad = True
        ins4.requires_grad = True

        van_real_out_ori, hr_feature     = D.forward_sp(hrs, need_feature=True)
        van_fake_out_ori, sr_feature     = D.forward_sp(srs, need_feature=True)
        van_ins16_out_ori, ins16_feature = D.forward_sp(ins16, need_feature=True)
        van_ins4_out_ori, ins4_feature   = D.forward_sp(ins4, need_feature=True)

        torch.mean(van_real_out_ori).backward()
        torch.mean(van_fake_out_ori).backward()
        torch.mean(van_ins16_out_ori).backward()
        torch.mean(van_ins4_out_ori).backward()


        block_0 = tensor_unit.kern_analyse_v3([x[0] for x in hr_feature],
                                              [x[0] for x in sr_feature],
                                              [x[0] for x in ins16_feature],
                                              [x[0] for x in ins4_feature],
                                              max_split_portion='average')
        
        block_1 = tensor_unit.kern_analyse_v3([x[1] for x in hr_feature],
                                              [x[1] for x in sr_feature],
                                              [x[1] for x in ins16_feature],
                                              [x[1] for x in ins4_feature],
                                              max_split_portion='average')

        
        # effectless   
        block_effectless_0 = block_0["effectless"]
        block_effectless_1 = block_1["effectless"]    

        for i in range(6):
            blist = [x for x in range(hr_feature[i][0].shape[1]) if x in block_effectless_0[str(i)]]
            kern_num = len(blist) + len(block_effectless_0[str(i)])
            bnum = min(len(block_effectless_0[str(i)]), len(blist), int(kern_num * block_portion))
            block_0[str(i)] = random.sample(blist, bnum)

        for i in range(6):
            blist = [x for x in range(hr_feature[i][1].shape[1]) if x in block_effectless_1[str(i)]]
            kern_num = len(blist) + len(block_effectless_1[str(i)])
            bnum = min(len(block_effectless_1[str(i)]), len(blist), int(kern_num * block_portion))
            block_1[str(i)] = random.sample(blist, bnum)

        block_effectless = {}
        for i in range(6):
            block_effectless[str(i)] = [block_0[str(i)], 
                                        block_1[str(i)]]  
            
        van_real_out = D.forward_sp(hrs.detach(),   dropout_index=block_effectless)
        van_fake_out = D.forward_sp(srs.detach(),   dropout_index=block_effectless)
        van_ins16_out= D.forward_sp(ins16.detach(), dropout_index=block_effectless)
        van_ins4_out = D.forward_sp(ins4.detach(),  dropout_index=block_effectless)

        hr_sr_relative.append(analyse_result(van_real_out, van_fake_out))
        sr_ins16_relative.append(analyse_result(van_fake_out, van_ins16_out))
        ins16_ins4_relative.append(analyse_result(van_ins16_out, van_ins4_out))


        rb_portion.append((len(block_effectless['0'][0]+block_effectless['0'][1]) / (64*2) + \
                           len(block_effectless['1'][0]+block_effectless['1'][1]) / (128*1.5) + \
                           len(block_effectless['2'][0]+block_effectless['2'][1]) / (256*1.5) + \
                           len(block_effectless['3'][0]+block_effectless['3'][1]) / (512*1.5) + \
                           len(block_effectless['4'][0]+block_effectless['4'][1]) / (512*2) + \
                           len(block_effectless['5'][0]+block_effectless['5'][1]) / (512*2))/6)
        

    print(sum(hr_sr_relative) / runtimes)
    print(sum(sr_ins16_relative) / runtimes)
    print(sum(ins16_ins4_relative) / runtimes)

    print(sum(rb_portion) / runtimes)


if __name__ == "__main__":
    RSGD_Test()

