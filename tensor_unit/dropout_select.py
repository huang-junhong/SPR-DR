import torch
import numpy as np


def compare_kern_effect(c1, c2, sort_type:str="abs", type:str="max"):

    diff = c1 - c2
    if sort_type == "abs":
        diff = torch.abs(diff)

    k_diff = torch.mean(diff, dim=[-1,-2])

    if type == "max":
        k_val, k_index = torch.sort(k_diff, dim=-1, descending=True)
    elif type == "min":
        k_val, k_index = torch.sort(k_diff, dim=-1, descending=False)

    return k_val, k_index


def get_KernGroup(feature_1, feature_2):

    assert len(feature_1) == len(feature_2)

    kerns = []
    for i in range(len(feature_1)):
        grad_1 = feature_1[i].grad
        grad_2 = feature_2[i].grad

        kerns.append(compare_kern_effect(grad_1, grad_2, sort_type="abs", type="max")[1])

    return kerns


def get_max_contribute_kern(index, portion=0.2, double_choose:bool=False):

    b, n = index.size()

    new_index = []
    
    vn = int(n*portion)

    temp = index[:, :vn].detach().cpu().numpy()
    element, count =  np.unique(temp, return_counts=True)

    new_index = []

    for e, c in zip(element, count):
        if c > b/2:
            new_index.append(e)


    if double_choose:
        vn = int((1-portion) * n)
        temp = index[:, vn:].detach().cpu().numpy()
        element, count =  np.unique(temp, return_counts=True)

        for e, c in zip(element, count):
            if c > b/2:
                new_index.append(e)

    return new_index


def split_group(group, split_mode="concrete", portion=0.5, reverse:bool=False):

    result = []

    if split_mode == "concrete":
        for now_group in group:
            if not reverse:
                result.append(get_max_contribute_kern(now_group, portion=portion))
            else:
                result.append(get_max_contribute_kern(torch.flip(now_group, [1]), portion=portion))
    elif split_mode == "average":
        pass

    return result


def get_block_kern(KernGroup_1, KernGroup_2, symbiosis:bool=False):
    
    result = []

    for kg1, kg2 in zip(KernGroup_1, KernGroup_2):
        if not symbiosis:
            temp = [x for x in kg1 if x not in kg2]
        else:
            temp = [x for x in kg1 if x in kg2]
        result.append(temp)

    return result


def combine_block_kern(group_1, group_2):
    result = {}

    for i in range(len(group_1)):
        result[i] = group_1[i] + group_2[i]

    return result


def analyse_result(judge1, judge2):

    relationship = torch.sigmoid(judge1 - judge2).detach().cpu().squeeze().numpy()

    connection = np.where(relationship > 0.5, 1, 0).flatten()

    c_portion = np.sum(connection) / connection.shape[0]

    return c_portion


def get_block_group(feature_hr, feature_sr, feature_ins16, feature_ins4):

    hr_sr_group = get_KernGroup(feature_hr, feature_sr)
    ins16_ins4_group = get_KernGroup(feature_ins16, feature_ins4)
    sr_ins16_group = get_KernGroup(feature_sr, feature_ins16)

    hr_sr_group_max = split_group(hr_sr_group)
    hr_sr_group_min = split_group(hr_sr_group, reverse=True)

    ins16_ins4_group_max = split_group(ins16_ins4_group)
    ins16_ins4_group_min = split_group(ins16_ins4_group, reverse=True)

    sr_ins16_group_max = split_group(sr_ins16_group)
    sr_ins16_group_min = split_group(sr_ins16_group, reverse=True)

    KG2KG1R = get_block_kern(ins16_ins4_group_min, hr_sr_group_min)
    KG3KG1R = get_block_kern(sr_ins16_group_min, hr_sr_group_min)

    KG1KG2 = get_block_kern(hr_sr_group_max, ins16_ins4_group_max)
    KG1KG3 = get_block_kern(hr_sr_group_max, sr_ins16_group_max)

    block_min = combine_block_kern(KG2KG1R, KG3KG1R)
    block_max = combine_block_kern(KG1KG2, KG1KG3)

    block_all = {}
    for i in range(6):
        block_all[str(i)] = list(set(block_max[i] + block_min[i]))

    return block_all
