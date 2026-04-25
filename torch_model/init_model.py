
import torch
import torch.optim as optim

from typing import Dict

from srres import SRRes2 as SRRes
from rrdn import RRDBNet, RRDBNet_SRF2
from real_rrdn import RRDBNet as Real_RRDBNet
from swinir import SwinIR

from Unet_Discriminator import Discriminator_UNet
from Stander_Discriminator_dynamic_dropout import Stander_Discriminator_with_dynamic_dropout


def init_swinir(model_config: Dict):
    srf = int(model_config.get('srf', 4))
    preset = model_config['generator']

    if preset == 'swinir-m':
        swinir_config = {
            'upscale': srf,
            'img_size': model_config.get('img_size', 64),
            'window_size': model_config.get('window_size', 8),
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': 'pixelshuffle',
            'resi_connection': '1conv',
        }
    elif preset == 'swinir-s':
        swinir_config = {
            'upscale': srf,
            'img_size': model_config.get('img_size', 64),
            'window_size': model_config.get('window_size', 8),
            'img_range': 1.,
            'depths': [6, 6, 6, 6],
            'embed_dim': 60,
            'num_heads': [6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': 'pixelshuffledirect',
            'resi_connection': '1conv',
        }
    else:
        raise ValueError(f'Unsupport SwinIR preset: {preset}')

    swinir_config.update(model_config.get('swinir_config', {}) or {})
    return SwinIR(**swinir_config)


def init_generator(model_config: Dict, train_config: Dict):
    """
    """

    #-------------------------------------------------------------
    # prepare model
    #-------------------------------------------------------------
    stric_load = True
    if model_config['generator'] == 'srres':
        model = SRRes()
    elif model_config['generator'] == 'rrdn':
        if model_config['srf'] == 2:
            model = RRDBNet_SRF2(3, 3, 64, 23)
        else:
            model = RRDBNet(3, 3, 64, 23)
    elif model_config['generator'] == 'real-rrdn':
        model = Real_RRDBNet(3, 3, model_config['srf'])
    elif model_config['generator'] in ['swinir-m', 'swinir-s']:
        model = init_swinir(model_config)
    else:
        raise ValueError(f'Unsupport generator type: {model_config["generator"]}')

    #-------------------------------------------------------------
    # load pretrian parameter
    #-------------------------------------------------------------
    if model_config['pretrain_path'] is not None:
        if model_config['model_key'] is None:
            model.load_state_dict({k.replace("module.", ""):v for k,v in torch.load(model_config['pretrain_path']).items()}, strict=stric_load)
        else:
            model.load_state_dict({k.replace("module.", ""):v for k,v in torch.load(model_config['pretrain_path'])[model_config['model_key']].items()}, strict=stric_load)

    #-------------------------------------------------------------
    # init optimizer 
    # init learning rate schedule
    #-------------------------------------------------------------
    if model_config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=float(model_config['init_lr']))

    if model_config['decay_schedule'] is None:
        lr_schedule = optim.lr_scheduler.ConstantLR(optimizer, 1., total_iters=train_config['train_iter'])
    elif model_config['decay_schedule']['mode'] == 'MultiStepLR':
        lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, model_config['decay_schedule']['decay_iter'], gamma=model_config['decay_schedule']['decay_weight'])
    else:
        raise ValueError(f'Unsupport learning rate schedual: {model_config["decay_schedule"]}')
    
    return model, optimizer, lr_schedule


def init_discriminator(model_config: Dict, train_config: Dict):
    """
    """

    #-------------------------------------------------------------
    # prepare model
    #-------------------------------------------------------------
    if model_config['discriminator'] == 'vgg-like':
        model = Stander_Discriminator_with_dynamic_dropout(normlize=model_config['normlize'])
    elif model_config['discriminator'] == 'unet':
        model = Discriminator_UNet(norm=model_config['normlize'])
    else:
        raise ValueError(f'Unsupport discriminator type: {model_config["discriminator"]}')

    #-------------------------------------------------------------
    # load pretrian parameter
    #-------------------------------------------------------------
    if model_config['pretrain_path'] is not None:
        if model_config['model_key'] is None:
            model.load_state_dict({k.replace("module.", ""):v for k,v in torch.load(model_config['pretrain_path']).items()}, strict=True)
        else:
            model.load_state_dict({k.replace("module.", ""):v for k,v in torch.load(model_config['pretrain_path'])[model_config['model_key']].items()}, strict=True)

    #-------------------------------------------------------------
    # init optimizer 
    # init learning rate schedule
    #-------------------------------------------------------------
    if model_config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=float(model_config['init_lr']))

    if model_config['decay_schedule'] is None:
        lr_schedule = optim.lr_scheduler.ConstantLR(optimizer, 1., total_iters=train_config['train_iter'])
    elif model_config['decay_schedule']['mode'] == 'MultiStepLR':
        lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, model_config['decay_schedule']['decay_iter'], gamma=model_config['decay_schedule']['decay_weight'])
    else:
        raise ValueError(f'Unsupport learning rate schedual: {model_config["decay_schedule"]}')
    
    return model, optimizer, lr_schedule


def init_model(model_config: Dict, train_config: Dict):
    """
    """

    if model_config.get('generator', None) is not None:
        return init_generator(model_config, train_config)
    elif model_config.get('discriminator', None) is not None:
        return init_discriminator(model_config, train_config)
    else:
        raise ValueError(f'No neccery key in {__name__}')
    
