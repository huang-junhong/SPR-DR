import torch

from typing import overload
from collections import Counter


def kern_effcet_split(contribute:torch.Tensor, max_split_portion=None, isabs:bool=True, sort_type:str="descend"):
    assert sort_type in ["descend", "ascend"]
    
    temp = contribute.clone().detach()
    
    if isabs:
        temp = torch.abs(temp)

    temp = torch.mean(temp, dim=[-2,-1])

    value, index = torch.sort(temp, dim=-1, descending=True if sort_type == "descend" else False)

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
        
        if sort_type == "descend":
            max_value = sorted_value[:, :split_pos]
            min_value = sorted_value[:, split_pos:]

            max_index = sorted_index[:, :split_pos]
            min_index = sorted_index[:, split_pos:]
        elif sort_type == "ascend":
            max_value = sorted_value[:, split_pos:]
            min_value = sorted_value[:, :split_pos]

            max_index = sorted_index[:, split_pos:]
            min_index = sorted_index[:, :split_pos]

        return max_value, min_value, max_index, min_index
    elif isinstance(max_split_portion, str):
        assert max_split_portion in ["average"]
        avg_value = torch.mean(sorted_value, dim=-1)

        max_value = []
        max_index = []

        min_value = []
        min_index = []

        for i in range(sorted_value.size(0)):

            if sort_type == "descend":
                _index = torch.where(sorted_value[i] > avg_value[i])[0] 

                if _index.size(0) == 0:
                    pos = int(sorted_value[i].size(0) * 0.75)
                else:
                    pos = _index[-1] + 1
                    
                max_value.append(sorted_value[i, :pos].tolist())
                max_index.append(sorted_index[i, :pos].tolist())

                min_value.append(sorted_value[i, pos:].tolist())
                min_index.append(sorted_index[i, pos:].tolist())
            elif sort_type == "ascend":
                _index = torch.where(sorted_value[i] > avg_value[i])[0]
                pos = _index[0]
            
                max_value.append(sorted_value[i, pos:].tolist())
                max_index.append(sorted_index[i, pos:].tolist())

                min_value.append(sorted_value[i, :pos].tolist())
                min_index.append(sorted_index[i, :pos].tolist())

        return max_value, min_value, max_index, min_index

def get_common_index(index1: list[list[int]]):

    result = set(index1[0])

    for x in index1:
        result = result & set(x)

    return result

def union_batch(batch_kern_index: list[list[int]]) -> set:

    union_set = None

    for indexs in batch_kern_index:
        if union_set is None:
            union_set = set(indexs)
        else:
            union_set = union_set | set(indexs)

    return union_set
    
def kern_analyse_v3(hr_feature, sr_feature, insk1_feature, insk2_feature, 
                    isabs=True, sort_type="descend", max_split_portion=0.5):
    
    result = {"effectless": {},
              "base_effect": {},
              "hs_effect": {},
              "sk1_effect": {},
              "k1k2_effect": {}}


    for i, (_hr_feature, _sr_feature, _insk1_feature, _insk2_feature) in\
        enumerate(zip(hr_feature, sr_feature, insk1_feature, insk2_feature)):
        
        #------------------------------------
        _hr_grad = _hr_feature.grad
        _sr_grad = _sr_feature.grad
        _insk1_grad = _insk1_feature.grad
        _insk2_grad = _insk2_feature.grad

        kern_num = _hr_feature.size(1)
        
        #------------------------------------
        # P1 split kern by feature grad
        #------------------------------------
        chr_value_ef, chr_value_efl, chr_index_ef, chr_index_efl = kern_effcet_split(_hr_grad, max_split_portion, isabs, sort_type)
        csr_value_ef, csr_value_efl, csr_index_ef, csr_index_efl = kern_effcet_split(_sr_grad, max_split_portion, isabs, sort_type)
        ck1_value_ef, ck1_value_efl, ck1_index_ef, ck1_index_efl = kern_effcet_split(_insk1_grad, max_split_portion, isabs, sort_type)
        ck2_value_ef, ck2_vlaue_efl, ck2_index_ef, ck2_index_efl = kern_effcet_split(_insk2_grad, max_split_portion, isabs, sort_type)

        #------------------------------------
        # P2 get effect-less-grad
        #------------------------------------
        chr_index_ef_set = union_batch(chr_index_ef)
        csr_index_ef_set = union_batch(csr_index_ef)
        ck1_index_ef_set = union_batch(ck1_index_ef)
        ck2_index_ef_set = union_batch(ck2_index_ef)

        effect_kern = chr_index_ef_set | csr_index_ef_set | ck1_index_ef_set | ck2_index_ef_set

        #-------------------------------------------------
        # P3 get channel contribute similar kern
        #-------------------------------------------------
        _, _, dc_hs_index_ef, dc_hs_index_efl  = compare_kern_effcet(_hr_feature, _sr_feature, max_split_portion, isabs, sort_type)
        _, _, dc_sk1_index_ef, dc_sk1_index_efl = compare_kern_effcet(_sr_feature, _insk1_feature, max_split_portion, isabs, sort_type)
        _, _, dc_k1k2_index_ef, dc_k1k2_index_efl = compare_kern_effcet(_insk1_feature, _insk2_feature, max_split_portion, isabs, sort_type)

        #
        dc_hs_index_ef_set   = union_batch(dc_hs_index_ef)
        dc_sk1_index_ef_set  = union_batch(dc_sk1_index_ef)
        dc_k1k2_index_ef_set = union_batch(dc_k1k2_index_ef)

        dc_effect_kern = dc_hs_index_ef_set | dc_sk1_index_ef_set | dc_k1k2_index_ef_set

        effect_kern = effect_kern | dc_effect_kern

        effectless_kern = set(range(kern_num)) - effect_kern

        result["effectless"][str(i)] = list(effectless_kern)

        #-------------------------------------------------
        # P4 Base effect
        #-------------------------------------------------
        base_effect = dc_hs_index_ef_set & dc_sk1_index_ef_set & dc_k1k2_index_ef_set
        result["base_effect"][str(i)] = list(base_effect) 

        #-------------------------------------------------
        # P5 specifical effect
        #-------------------------------------------------
        hs_effect_set = chr_index_ef_set & csr_index_ef_set - base_effect
        
        sk1_effect_set = csr_index_ef_set & ck1_index_ef_set - base_effect

        k1k2_effect_set = ck1_index_ef_set & ck2_index_ef_set - base_effect

        gcs = hs_effect_set | sk1_effect_set | k1k2_effect_set | base_effect

        result["k1k2_effect"][str(i)] = list(gcs)

        result["hs_effect"][str(i)] = list(effect_kern)

        ...

    return result



