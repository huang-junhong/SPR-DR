import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r1: torch.Tensor, r2: torch.Tensor, is_r1_bettter: bool = True, 
                sigmoid:bool = True, margin: float = 0, reduction: str = 'mean'):
        if is_r1_bettter:
            gamma = torch.ones_like(r1)
        else:
            gamma = -1 * torch.ones_like(r1)

        
        if not sigmoid:
            return F.margin_ranking_loss(r1, r2, gamma, margin=margin, reduction=reduction)
        else:
            return F.margin_ranking_loss(F.sigmoid(r1), F.sigmoid(r2), gamma, margin=margin, reduction=reduction)