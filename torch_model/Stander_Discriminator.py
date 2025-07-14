import torch.nn as nn


class Stander_Discriminator(nn.Module):
    def __init__(self, indim:int=3, wkdim:int=64):
        super(Stander_Discriminator,self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv2d(indim, wkdim, 3, 1, 1),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(wkdim, wkdim, 4, 2, 1),
                                     nn.InstanceNorm2d(wkdim),
                                     nn.LeakyReLU())
        
        self.layer_2 = nn.Sequential(nn.Conv2d(wkdim, wkdim*2, 3, 1, 1),
                                     nn.InstanceNorm2d(wkdim*2),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(wkdim*2, wkdim*2, 4, 2, 1),
                                     nn.InstanceNorm2d(wkdim*2),
                                     nn.LeakyReLU())
        
        self.layer_3 = nn.Sequential(nn.Conv2d(wkdim*2, wkdim*4, 3, 1, 1),
                                     nn.InstanceNorm2d(wkdim*4),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(wkdim*4, wkdim*4, 4, 2, 1),
                                     nn.InstanceNorm2d(wkdim*4),
                                     nn.LeakyReLU())
        
        self.layer_4 = nn.Sequential(nn.Conv2d(wkdim*4, wkdim*8, 3, 1, 1),
                                     nn.InstanceNorm2d(wkdim*8),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(wkdim*8, wkdim*8, 4, 2, 1),
                                     nn.InstanceNorm2d(wkdim*8),
                                     nn.LeakyReLU())
        
        self.layer_5 = nn.Sequential(nn.Conv2d(wkdim*8, wkdim*8, 3, 1, 1),
                                     nn.InstanceNorm2d(wkdim*8),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(wkdim*8, wkdim*8, 4, 2, 1),
                                     nn.InstanceNorm2d(wkdim*8),
                                     nn.LeakyReLU())
        
        self.tail    = nn.Sequential(nn.Conv2d(wkdim*8,wkdim*8,4,2,1),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(wkdim*8,wkdim*8,3,1,1),
                                     nn.LeakyReLU(),
                                     nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Conv2d(wkdim*8,1,1,1,0))
        
        
    def forward(self, input, need_feature=False):
        current = self.layer_1(input)
        current = self.layer_2(current)
        current = self.layer_3(current)
        if need_feature:
            return current.detach()
        current = self.layer_4(current)
        current = self.layer_5(current)
        current = self.tail(current)
        current = current.squeeze(1)
        return current


