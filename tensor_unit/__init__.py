import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ChannelDropout import ChannelDropout
from dropout_select import get_block_group
from Patch_Replacement import Replacement, Replacement_v2
from dropout_select_v2 import kern_analyse, kern_select_by_discriminator_optimize
from dropout_select_v3 import kern_analyse_v3
from init_spr_module import init_spr_module