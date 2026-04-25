from typing import Dict
from torch.utils.data import DataLoader


def init_dataloader(data_config: Dict):
    """
    """

    if data_config['mode'] is None or\
       data_config['mode'] == 'synthetic':
        from synthetic_dataloader import SyntheticDataset
        from synthetic_dataloader import data_prefetcher
        dataset = SyntheticDataset(data_config['trainset'])
        dataloader = DataLoader(dataset, batch_size=data_config["batch_size"], num_workers=data_config["batch_size"]//4, shuffle=True, pin_memory=True, drop_last=True, prefetch_factor=4)
        prefecher = data_prefetcher(dataloader)
    elif data_config['mode'] == 'realesrgan':
        from real_esrgan_dataloader import RealESRGANDataset
        from real_esrgan_dataloader import data_prefetcher as real_esragn_prefetcher
        dataset = RealESRGANDataset(data_config['trainset'])
        dataloader = DataLoader(dataset, batch_size=data_config["batch_size"], num_workers=data_config["batch_size"]//4, shuffle=True, pin_memory=True, drop_last=True, prefetch_factor=4)
        prefecher = real_esragn_prefetcher(dataloader, data_config['hr_size'], data_config['srf'])
    elif data_config['mode'] == 'realesrgan-c1':
        from real_esrgan_dataloader_c1 import RealESRGANDataset
        from real_esrgan_dataloader_c1 import data_prefetcher as real_esragn_prefetcher
        dataset = RealESRGANDataset(data_config['trainset'])
        dataloader = DataLoader(dataset, batch_size=data_config["batch_size"], num_workers=data_config["batch_size"]//4, shuffle=True, pin_memory=True, drop_last=True, prefetch_factor=4)
        prefecher = real_esragn_prefetcher(dataloader, data_config['hr_size'], data_config['srf'], need_l1=data_config['need_l1'])
    elif data_config['mode'] == 'mri':
        from mri_synthetic_dataloader import MRISyntheticDataset
        from mri_synthetic_dataloader import data_prefetcher as mri_prefecher
        dataset = MRISyntheticDataset(data_config['trainset'])
        dataloader = DataLoader(dataset, batch_size=data_config["batch_size"], num_workers=data_config["batch_size"]//4, shuffle=True, pin_memory=True, drop_last=True, prefetch_factor=4)
        prefecher = mri_prefecher(dataloader)
    else:
        raise ValueError(f'Unsupport dataloader mode: {data_config["mode"]}')
    return dataloader, prefecher
