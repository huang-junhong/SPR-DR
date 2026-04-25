from typing import Dict

from Patch_Replacement import Replacement, Replacement_v2


def init_spr_module(train_config: Dict):
    """
    """

    hr_size = train_config['data_config']['hr_size']
    if isinstance(hr_size, int):
        hr_size = (hr_size, hr_size)

    spr_config: Dict = train_config['spr_config']

    spr_config_key = [key for key in spr_config.keys() if key.startswith('ranker_')]

    sprs = []

    for idx, key in enumerate(spr_config_key):
        
        _sprs = []
        for r_size in spr_config[key]:
            rep = Replacement_v2(r_size=r_size, size=hr_size, roll_prob=spr_config['roll_prob'])
            _sprs.append(rep)

        sprs.append(_sprs)

    return sprs
