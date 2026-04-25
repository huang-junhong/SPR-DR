import torch
import torch.nn as nn


class D_LOSS(nn.Module):
    def __init__(self):
        super(D_LOSS, self).__init__()
        self.BCE_LOSS = nn.BCEWithLogitsLoss()

    def forward(self, input, target, weight=5e-3, type='vanilla'):
        if type == 'vanilla':
            loss = self.BCE_LOSS(input, target)
        elif type == 'relative_opt_d':
            loss = torch.mean(self.BCE_LOSS(input-target, torch.zeros_like(input)) +
                              self.BCE_LOSS(target-input, torch.ones_like(target)))
        elif type == 'relative_opt_g':
            loss = torch.mean(self.BCE_LOSS(target-input, torch.zeros_like(target)) +
                              self.BCE_LOSS(input-target, torch.ones_like(input)))
        return loss * weight