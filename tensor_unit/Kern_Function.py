import torch
import numpy as np

from typing import overload


def kern_effcet_split(contribute:torch.Tensor, max_split_portion=None, isabs:bool=True, sort_type:str="descend"):
    assert sort_type in ["descend", "ascend"]
    
    temp = contribute.clone().detach()
    
    if isabs:
        temp = torch.abs(temp)

    temp = torch.mean(temp, dim=[-2,-1])

    value, index = torch.sort(temp, descending=True if sort_type == "descend" else False)


    return split_kern_by_effect(value, index, max_split_portion, sort_type)


def get_kern_effect_difference(c1, c2, isabs:bool=True, sort_type:str=None):

    assert sort_type in [None, "descend", "ascend"]

    diff = c1 - c2

    if isabs:
        diff = torch.abs(diff)

    diff = torch.mean(diff, dim=[-1,-2])

    if sort_type is None:
        return diff, None
    elif sort_type == "ascend":
        k_val, k_index = torch.sort(diff, dim=-1, descending=False)
        return k_val, k_index
    else:
        k_val, k_index = torch.sort(diff, dim=-1, descending=True)
        return k_val, k_index


def compare_kern_effcet(c1, c2, max_split_portion, isabs:bool=True, sort_type:str=None):
    
    value, index = get_kern_effect_difference(c1, c2, isabs, sort_type)

    return split_kern_by_effect(value, index, max_split_portion, sort_type)


@overload
def split_kern_by_effect(sorted_value, sorted_index, max_split_portion:float, sort_type:str="descent") ->\
      tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pass
@overload
def split_kern_by_effect(sorted_value, sorted_index, max_split_portion:str, sort_type:str="descent") ->\
      tuple[list, list, list, list]:
    pass

def split_kern_by_effect(sorted_value, sorted_index, max_split_portion=0.5, sort_type:str="descent"):

    if isinstance(max_split_portion, float):
        assert max_split_portion > 0 and max_split_portion < 1
        all_num = sorted_index.size(1)
        split_pos = int(all_num * max_split_portion)
        max_value = sorted_value[:,:split_pos]
        min_value = sorted_value[:,split_pos:]

        max_index = sorted_index[:,:split_pos]
        min_index = sorted_index[:,split_pos:]
        return max_value, min_value, max_index, min_index
    elif isinstance(max_split_portion, str):
        assert max_split_portion in ["average"]
        avg_value = torch.mean(sorted_value, dim=-1)

        max_index = []
        max_value = []
        min_index = []
        min_value = []
        
        for i in range(avg_value.size(0)):
            _avg_value = avg_value[i]
            if sort_type == "descend":
                pos = torch.where(sorted_value[i] > _avg_value)[0][-1]

                max_index.append(sorted_index[i, :pos].tolist())
                max_value.append(sorted_value[i, :pos].tolist())

                min_index.append(sorted_index[i, pos:].tolist())
                min_value.append(sorted_value[i, pos:].tolist())
            elif sort_type == "ascend":
                pos = torch.where(sorted_value[i] > _avg_value)[0][0]
                
                min_index.append(sorted_index[i, :pos])
                min_value.append(sorted_value[i, :pos])

                max_index.append(sorted_index[i, pos:])
                max_value.append(sorted_value[i, pos:])

        return max_value, min_value, max_index, min_index


def kern_analyse(self, hr_feature, sr_feature, insk1_feature, insk2_feature, 
                 isabs=True, sort_type="descend", max_split_portion="average"):
    kern_contribute = {}

    for _hr_feature, _sr_feature, _insk1_feature, _insk2_feature in\
        zip(hr_feature, sr_feature, insk1_feature, insk2_feature):
        
        #------------------------------------
        _hr_grad = _hr_feature.grad
        _sr_grad = _sr_feature.grad
        _insk1_grad = _insk1_feature.grad
        _insk2_grad = _insk2_feature.grad

        kern_num = _hr_feature.size(1)
        
        #------------------------------------
        chr_value, chr_index = kern_effcet_split(_hr_grad, max_split_portion, isabs, sort_type)
        csr_value, csr_index = kern_effcet_split(_sr_grad, max_split_portion, isabs, sort_type)
        ck1_value, ck1_index = kern_effcet_split(_insk1_grad, max_split_portion, isabs, sort_type)
        ck2_value, ck2_index = kern_effcet_split(_insk2_grad, max_split_portion, isabs, sort_type)

        c_effect_kern = list(set(chr_index + csr_index + ck1_index + ck2_index))
        c_effectless_kern = [x for x in range(kern_num) if x not in c_effect_kern]

        #------------------------------------
        dchs_value, dchs_index = compare_kern_effcet(_hr_grad, _sr_grad, max_split_portion, isabs, sort_type)
        dcsk1_value, dcsk1_index = compare_kern_effcet(_sr_grad, _insk1_grad, max_split_portion, isabs, sort_type)
        dck1k2_value, dck1k2_index = compare_kern_effcet(_insk1_grad, _insk2_grad, max_split_portion, isabs, sort_type)


if __name__ == "__main__":

    pass