import torch
import torchvision
import torch.nn as nn

from typing import List


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm

        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)]).cuda()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class Multi_VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2, 7, 17, 25, 34], use_bn=False, use_input_norm=True):
        super(Multi_VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm

        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True).cuda()
        else:
            model = torchvision.models.vgg19(pretrained=True).cuda()

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)


        self.features = []
        for i in range(len(feature_layer)):
            if i == 0:
                self.features.append(nn.Sequential(*list(model.features.children())[:(feature_layer[i] + 1)]))
            else:
                self.features.append(nn.Sequential(*list(model.features.children())[(feature_layer[i-1]+1):(feature_layer[i] + 1)]))
        # No need to BP to variable
        for i in range(len(feature_layer)):
            for k, v in self.features[i].named_parameters():
                v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]

        loss = 0.

        features = []
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        for i in range(len(self.features)):
            x = self.features[i](x)
            features.append(x)

        return features
    

class Perceptual_Loss(nn.Module):
    def __init__(self, feature_layers: List[int]):
        super(Perceptual_Loss, self).__init__()
        self.distance_measure = nn.L1Loss()
        self.feature_extractor = VGGFeatureExtractor().cuda()
        self.multi_feature_extractor = Multi_VGGFeatureExtractor(feature_layers).cuda()

    def forward(self, hr, sr, weight=1.):

        if isinstance(weight, (float,int)):
            real_feature = self.feature_extractor(hr)
            fake_feature = self.feature_extractor(sr)
            loss = self.distance_measure(fake_feature, real_feature) * weight

        elif isinstance(weight, list):
            real_feature = self.multi_feature_extractor(hr)
            fake_feature = self.multi_feature_extractor(sr)
            loss = 0.
            for i in range(len(weight)):
                loss += self.distance_measure(fake_feature[i], real_feature[i]) * weight[i]

        return loss
    