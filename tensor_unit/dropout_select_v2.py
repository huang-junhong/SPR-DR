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

    return list(set(result))
    
def kern_analyse(hr_feature, sr_feature, insk1_feature, insk2_feature, 
                 isabs=True, sort_type="descend", max_split_portion=0.5):
    
    kern_contribute = {}

    effectless_index = {}

    all_effect_index = {}


    hs_both_effect_index = {}
    sk1_both_effect_index = {}
    k1k2_both_effect_index = {}


    for i, (_hr_feature, _sr_feature, _insk1_feature, _insk2_feature) in\
        enumerate(zip(hr_feature, sr_feature, insk1_feature, insk2_feature)):
        
        #------------------------------------
        _hr_grad = _hr_feature.grad
        _sr_grad = _sr_feature.grad
        _insk1_grad = _insk1_feature.grad
        _insk2_grad = _insk2_feature.grad

        kern_num = _hr_feature.size(1)
        
        #------------------------------------
        chr_value_ef, chr_value_efl, chr_index_ef, chr_index_efl = kern_effcet_split(_hr_grad, max_split_portion, isabs, sort_type)
        csr_value_ef, csr_value_efl, csr_index_ef, csr_index_efl = kern_effcet_split(_sr_grad, max_split_portion, isabs, sort_type)
        ck1_value_ef, ck1_value_efl, ck1_index_ef, ck1_index_efl = kern_effcet_split(_insk1_grad, max_split_portion, isabs, sort_type)
        ck2_value_ef, ck2_vlaue_efl, ck2_index_ef, ck2_index_efl = kern_effcet_split(_insk2_grad, max_split_portion, isabs, sort_type)
        
        effect_kern = []
        hs_effect_kern = []
        sk1_effect_kern = []
        k1k2_effect_kern = []

        hs_both_effect_kern = []

        for hr_ef, sr_ef, insk1_ef, insk2_ef in zip(chr_index_ef, csr_index_ef, ck1_index_ef, ck2_index_ef):
            if isinstance(max_split_portion, float):
                temp = list(set(hr_ef.tolist() + sr_ef.tolist() + insk1_ef.tolist() + insk2_ef.tolist()))
                temp_hs = list(set(hr_ef.tolist() + sr_ef.tolist()))
                temp_sk1 = list(set(sr_ef.tolist() + insk1_ef.tolist()))
                temp_k1k2 = list(set(insk1_ef.tolist() + insk2_ef.tolist()))
                temp_basic_kern = [x for x in hr_ef.tolist() if x in sr_ef.tolist() and x in insk1_ef.tolist() and x in insk2_ef.tolist()]
            else:
                temp = list(set(hr_ef + sr_ef + insk1_ef + insk2_ef))
                temp_hs = list(set(hr_ef + sr_ef))
                temp_sk1 = list(set(sr_ef + insk1_ef))
                temp_k1k2 = list(set(insk1_ef + insk2_ef))

                temp_hrb = [x for x in hr_ef if x in sr_ef]
            effect_kern.extend(temp)
            hs_effect_kern.extend(temp_hs)
            sk1_effect_kern.extend(temp_sk1)
            k1k2_effect_kern.extend(temp_k1k2)
            hs_both_effect_kern.extend(temp_hrb)
        
        effect_kern = list(set(effect_kern))
        hs_effect_kern = list(set(hs_effect_kern))
        sk1_effect_kern = list(set(sk1_effect_kern))
        k1k2_effect_kern = list(set(k1k2_effect_kern))
        hs_both_effect_kern = list(set(hs_both_effect_kern))

        effectless_kern = [x for x in range(kern_num) if x not in effect_kern]

        #------------------------------------
        _, _, dc_hs_index_ef, dc_hs_index_efl  = compare_kern_effcet(_hr_feature, _sr_feature, max_split_portion, isabs, sort_type)
        _, _, dc_sk1_index_ef, dc_sk1_index_efl = compare_kern_effcet(_sr_feature, _insk1_feature, max_split_portion, isabs, sort_type)
        _, _, dc_k1k2_index_ef, dc_k1k2_index_efl = compare_kern_effcet(_insk1_feature, _insk2_feature, max_split_portion, isabs, sort_type)

        dc_effect_kern = []

        hs_dmax = []
        sk1_dmax = []
        k1k2_dmax = []

        for dc_hs_ef, dc_sk1_ef, dc_k1k2_ef in zip(dc_hs_index_ef, dc_sk1_index_ef, dc_k1k2_index_ef):
            if isinstance(max_split_portion, float):
                temp = list(set(dc_hs_ef.tolist() + dc_sk1_ef.tolist() + dc_k1k2_ef.tolist()))
                temp_hs = list(set(dc_hs_ef.tolist()))
                temp_sk1 = list(set(dc_sk1_ef.tolist()))
                temp_k1k2 = list(set(dc_k1k2_index_ef.tolist()))
            else:
                temp = list(set(dc_hs_ef + dc_sk1_ef + dc_k1k2_ef))
                temp_hs = list(set(dc_hs_ef))
                temp_sk1 = list(set(dc_sk1_ef))
                temp_k1k2 = list(set(dc_k1k2_ef))
            dc_effect_kern.extend(temp)
            hs_dmax.extend(temp_hs)
            sk1_dmax.extend(temp_sk1)
            k1k2_dmax.extend(temp_k1k2)

        dc_effect_kern = list(set(dc_effect_kern))
        hs_dmax = list(set(hs_dmax))
        sk1_dmax = list(set(sk1_dmax))
        k1k2_dmax = list(set(k1k2_dmax))

        dc_effectless_kern = [x for x in range(kern_num) if x not in dc_effect_kern]
        
        #
        effectless_index[str(i)] = [x for x in effectless_kern if x in dc_effectless_kern]

        # all-effect
        hs_both_effect_index[str(i)] = [x for x in hs_both_effect_kern if x in hs_dmax]

        # sk1-both-effect
        sk1_both_effect_index[str(i)] = [x for x in sk1_effect_kern if x in sk1_dmax]

        # k1k2-both-effect
        k1k2_both_effect_index[str(i)] = [x for x in k1k2_effect_kern if x in k1k2_dmax]

        # all-effect
        all_effect_index[str(i)] = [x for x in hs_both_effect_index[str(i)] if x in sk1_both_effect_index[str(i)] and x in k1k2_both_effect_index[str(i)] and\
                                          x in hs_dmax and x in sk1_dmax and x in k1k2_dmax]

    kern_contribute["effectless"]  = effectless_index
    kern_contribute["hs_effect"]   = hs_both_effect_index
    kern_contribute["sk1_effect"]  = sk1_both_effect_index
    kern_contribute["k1k2_effect"] = k1k2_both_effect_index
    kern_contribute["all_effect"]  = all_effect_index

    
    return kern_contribute


def kern_select_by_discriminator_optimize(hr, sr, insk1, insk2, D, D_OPT, discriminator_loss, portion="avg"):
    
    D_OPT.zero_grad()
    D.train()

    van_real_out= D(hr)
    van_fake_out = D(sr.detach())
    van_insk1_out = D(insk1)
    van_insk2_out = D(insk2)

    dvan_loss = discriminator_loss(van_fake_out, van_real_out, 1, "relative_opt_d") +\
                discriminator_loss(van_insk1_out, van_fake_out, 1, "relative_opt_d") +\
                discriminator_loss(van_insk2_out, van_insk1_out, 1, "relative_opt_d")
    
    dvan_loss.backward()

    index = {}

    #layer 0
    layer0_grad = torch.mean(torch.abs(D.layer_0.layer[0].weight.grad), dim=[1,2,3])
    vl0, il0 = torch.sort(layer0_grad, descending=True)
    il0 = il0.tolist()

    #layer 1
    layer11_grad = torch.mean(torch.abs(D.layer_1.layer[0].weight.grad), dim=[1,2,3])
    layer12_grad = torch.mean(torch.abs(D.layer_1.layer[3].weight.grad), dim=[1,2,3])

    vl11, il11 = torch.sort(layer11_grad, descending=True)
    vl12, il12 = torch.sort(layer12_grad, descending=True)

    il11 = il11.tolist()
    il12 = il12.tolist()

    #layer 2
    layer21_grad = torch.mean(torch.abs(D.layer_2.layer[0].weight.grad), dim=[1,2,3])
    layer22_grad = torch.mean(torch.abs(D.layer_2.layer[3].weight.grad), dim=[1,2,3])

    vl21, il21 = torch.sort(layer21_grad, descending=True)
    vl22, il22 = torch.sort(layer22_grad, descending=True)

    il21 = il21.tolist()
    il22 = il22.tolist()

    #layer 3
    layer31_grad = torch.mean(torch.abs(D.layer_3.layer[0].weight.grad), dim=[1,2,3])
    layer32_grad = torch.mean(torch.abs(D.layer_3.layer[3].weight.grad), dim=[1,2,3])

    vl31, il31 = torch.sort(layer31_grad, descending=True)
    vl32, il32 = torch.sort(layer32_grad, descending=True)

    il31 = il31.tolist()
    il32 = il32.tolist()

    #layer 4
    layer41_grad = torch.mean(torch.abs(D.layer_4.layer[0].weight.grad), dim=[1,2,3])
    layer42_grad = torch.mean(torch.abs(D.layer_4.layer[3].weight.grad), dim=[1,2,3])

    vl41, il41 = torch.sort(layer41_grad, descending=True)
    vl42, il42 = torch.sort(layer42_grad, descending=True)

    il41 = il41.tolist()
    il42 = il42.tolist()

    #layer 5
    layer51_grad = torch.mean(torch.abs(D.layer_5.layer[0].weight.grad), dim=[1,2,3])
    layer52_grad = torch.mean(torch.abs(D.layer_5.layer[3].weight.grad), dim=[1,2,3])

    vl51, il51 = torch.sort(layer51_grad, descending=True)
    vl52, il52 = torch.sort(layer52_grad, descending=True)

    il51 = il51.tolist()
    il52 = il52.tolist()

    if isinstance(portion, float):
        il0_num = len(il0)
        index[str(0)] = il0[-int(il0_num*portion):]

        il1_num = len(il11)
        index[str(1)] = [il11[-int(il1_num*portion):],
                         il12[-int(il1_num*portion):]]
        
        il2_num = len(il21)
        index[str(2)] = [il21[-int(il2_num*portion):],
                         il22[-int(il2_num*portion):]]
        
        il3_num = len(il31)
        index[str(3)] = [il31[-int(il3_num*portion):],
                         il32[-int(il3_num*portion):]]
        
        il4_num = len(il41)
        index[str(4)] = [il41[-int(il4_num*portion):],
                         il42[-int(il4_num*portion):]]
        
        il5_num = len(il51)
        index[str(5)] = [il51[-int(il5_num*portion):],
                         il52[-int(il5_num*portion):]]

    return index


def find_max_difference(lists, topcount):
    all_elements = [element for lst in lists for element in lst]
    element_freq = Counter(all_elements)

    unique_score = [(i, lst, sum(1/element_freq[element] for element in lst)) for i, lst in enumerate(lists)]

    sorted_lists = sorted(unique_score, key=lambda x: x[2], reverse=True)

    return [index for index, _, _ in sorted_lists[:topcount]]





