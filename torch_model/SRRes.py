import torch.nn as nn


class ResUnit(nn.Module):
    def __init__(self, ksize=3, wkdim=64):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize/2))
        self.active = nn.PReLU()
        self.conv2 = nn.Conv2d(wkdim, wkdim, ksize, 1, int(ksize/2))

    def forward(self, input):
        current = self.conv1(input)
        current = self.active(current)
        current = self.conv2(current)
        current = input + current
        return current


class UPN(nn.Module):
    def __init__(self, indim=64, scale=2):
        super(UPN, self).__init__()
        self.conv = nn.Conv2d(indim, indim*(scale**2), 3, 1, 1)
        self.Upsample = nn.PixelShuffle(scale)
        self.active = nn.PReLU()

    def forward(self, input):
        current = self.conv(input)
        current = self.Upsample(current)
        current = self.active(current)
        return current


class SRRes(nn.Module):
    def __init__(self, wkdim=64, num_block=16):
        super(SRRes, self).__init__()
        self.head = nn.Conv2d(3, wkdim, 9, 1, 4)
        self.resblock = self._make_resblocks(wkdim, num_block)
        self.gate = nn.Conv2d(wkdim, wkdim, 3, 1, 1)
        self.up_1 = UPN(wkdim)
        self.up_2 = UPN(wkdim)

        self.comp = nn.Conv2d(wkdim*2, wkdim, 3, 1, 1)

        self.tail = nn.Conv2d(wkdim, 3, 9, 1, 4)

    def _make_resblocks(self, wkdim, num_block):
        layers = []
        for i in range(1, num_block+1):
            layers.append(ResUnit(wkdim=wkdim))
        return nn.Sequential(*layers)

    def forward(self, input):
        F_0 = self.head(input)
        current = self.resblock(F_0)
        current = self.gate(current)
        current = F_0 + current
        UP_1 = self.up_1(current)
        UP_2 = self.up_2(UP_1)

        current = self.tail(UP_2)
        return current
    
class SRRes_SRF2(nn.Module):
    def __init__(self, wkdim=64, num_block=16):
        super(SRRes_SRF2, self).__init__()
        self.head = nn.Conv2d(3, wkdim, 9, 1, 4)
        self.resblock = self._make_resblocks(wkdim, num_block)
        self.gate = nn.Conv2d(wkdim, wkdim, 3, 1, 1)
        self.up_1 = UPN(wkdim)

        self.comp = nn.Conv2d(wkdim*2, wkdim, 3, 1, 1)

        self.tail = nn.Conv2d(wkdim, 3, 9, 1, 4)

    def _make_resblocks(self, wkdim, num_block):
        layers = []
        for i in range(1, num_block+1):
            layers.append(ResUnit(wkdim=wkdim))
        return nn.Sequential(*layers)

    def forward(self, input):
        F_0 = self.head(input)
        current = self.resblock(F_0)
        current = self.gate(current)
        current = F_0 + current
        UP_1 = self.up_1(current)

        current = self.tail(UP_1)
        return current