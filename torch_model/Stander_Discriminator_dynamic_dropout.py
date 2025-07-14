import torch
import torch.nn as nn

from typing import Union

from tensor_unit import ChannelDropout


class head_layer_A(nn.Module):
    def __init__(self, indim:int=3, wkdim:int=64) -> None:
        super(head_layer_A, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(indim, wkdim, 3, 1, 1),
                                   nn.LeakyReLU())
        self.dropout = ChannelDropout()
        
    def forward(self, input, dropout_index):
        temp = self.layer(input)
        temp = self.dropout(temp, dropout_index)
        return temp
    

class base_layer_A(nn.Module):
    def __init__(self, indim:int, wkdim:int, outdim:Union[None,int]=None) -> None:
        super(base_layer_A, self).__init__()

        if outdim is None:
            outdim = int(wkdim * 2)
        self.layer = nn.Sequential(nn.Conv2d(indim, wkdim, 4, 2, 1),
                                   nn.InstanceNorm2d(wkdim),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(wkdim, outdim, 3, 1, 1),
                                   nn.InstanceNorm2d(outdim),
                                   nn.LeakyReLU(),)
        self.dropout = ChannelDropout()


    def forward_sp(self, input, dropout_index_1, dropout_index_2):
        temp = self.layer[0](input)
        temp = self.layer[1](temp)
        temp = self.layer[2](temp)
        feature_1 = self.dropout(temp, dropout_index_1)

        temp = self.layer[3](feature_1)
        temp = self.layer[4](temp)
        temp = self.layer[5](temp)
        feature_2 = self.dropout(temp, dropout_index_2)

        return feature_1, feature_2


    def forward(self, input, dropout_index):
        layer_feature = self.layer(input)
        droped_feature = self.dropout(layer_feature, dropout_index)
        return droped_feature
    

class tail_layer_A(nn.Module):
    def __init__(self, indim:int, outdim:int=1) -> None:
        super(tail_layer_A, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(indim, indim, 3, 1, 1),
                                   nn.LeakyReLU(),
                                   nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(indim, outdim, 1, 1, 0))
        

    def forward(self, input):
        return self.layer(input)
        


class Stander_Discriminator_with_dynamic_dropout(nn.Module):
    def __init__(self, indim:int=3, wkdim:int=64) -> None:
        super(Stander_Discriminator_with_dynamic_dropout, self).__init__()
        self.layer_0 = head_layer_A(indim, wkdim)
        self.layer_1 = base_layer_A(wkdim, wkdim)
        self.layer_2 = base_layer_A(wkdim*2, wkdim*2)
        self.layer_3 = base_layer_A(wkdim*4, wkdim*4)
        self.layer_4 = base_layer_A(wkdim*8, wkdim*8, wkdim*8)
        self.layer_5 = base_layer_A(wkdim*8, wkdim*8, wkdim*8)
        self.layer_6 = tail_layer_A(wkdim*8)


    def forward_sp(self, input, need_feature:bool=False, dropout_index:Union[dict,None]=None):
        if dropout_index is None:
            dropout_index = {}
            for i in range(6):
                dropout_index[str(i)] = [[], []]

        f0 = self.layer_0(input, dropout_index['0'])
        f10, f11 = self.layer_1.forward_sp(f0, dropout_index['1'][0], dropout_index['1'][1])
        f20, f21 = self.layer_2.forward_sp(f11, dropout_index['2'][0], dropout_index['2'][1])
        f30, f31 = self.layer_3.forward_sp(f21, dropout_index['3'][0], dropout_index['3'][1])
        f40, f41 = self.layer_4.forward_sp(f31, dropout_index['4'][0], dropout_index['4'][1])
        f50, f51 = self.layer_5.forward_sp(f41, dropout_index['5'][0], dropout_index['5'][1])
        f6 = self.layer_6(f51)

        if need_feature:
            f0.retain_grad()
            f10.retain_grad()
            f11.retain_grad()
            f20.retain_grad()
            f21.retain_grad()
            f30.retain_grad()
            f31.retain_grad()
            f40.retain_grad()
            f41.retain_grad()
            f50.retain_grad()
            f51.retain_grad()
            return f6, [[f0, f0],[f10, f11],[f20, f21], [f30, f31], [f40, f41],[f50, f51]]
        else:
            return f6


    def forward(self, input, need_feature:bool=False, dropout_index:Union[dict,None]=None):
        if dropout_index is None:
            dropout_index = {}
            for i in range(6):
                dropout_index[str(i)] = []

        f0 = self.layer_0(input, dropout_index['0'])
        f1 = self.layer_1(f0, dropout_index['1'])
        f2 = self.layer_2(f1, dropout_index['2'])
        f3 = self.layer_3(f2, dropout_index['3'])
        f4 = self.layer_4(f3, dropout_index['4'])
        f5 = self.layer_5(f4, dropout_index['5'])
        f6 = self.layer_6(f5)

        if need_feature:
            f0.retain_grad()
            f1.retain_grad()
            f2.retain_grad()
            f3.retain_grad()
            f4.retain_grad()
            f5.retain_grad()
            features = [f0, f1, f2, f3, f4, f5]
            return f6, features
        else:
            return f6

