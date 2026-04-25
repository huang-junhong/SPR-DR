import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from gan_loss import D_LOSS
from ssim_loss import SSIM
from rank_loss import MarginRankingLoss
from perceptual_loss import Perceptual_Loss
from spr_perceptual_loss import SPRPerceptualLoss