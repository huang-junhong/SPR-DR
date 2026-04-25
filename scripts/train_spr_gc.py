import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["TRAIN-ROOT-PATH"] = str(PROJECT_ROOT)

import torch
import pyiqa
import torch.nn as nn

import loss
import file_io
import functional
import tensor_unit
import torch_model
import data_module

from loguru import logger
from datetime import datetime
from progress.bar import FillingSquaresBar
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPR with gradient accumulation.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "train_spr_example.yaml"),
        help="Path to a YAML training config.",
    )
    return parser.parse_args()


def init_tensorboard(train_configs, config_path):
    tensorboard_config = train_configs.get("tensorboard", {})
    log_dir = tensorboard_config.get("log_dir") or train_configs["validation"]["save_root"]
    writer = SummaryWriter(log_dir=log_dir)
    with open(config_path, "r", encoding="utf-8") as file:
        writer.add_text("config/train_yaml", f"```yaml\n{file.read()}\n```", global_step=0)
    return writer


def log_validation(writer, eval_results, global_step):
    for result in eval_results:
        dataset = result["dataset"]
        for key, value in result.items():
            if key != "dataset":
                writer.add_scalar(f"validation/{dataset}/{key}", value, global_step=global_step)


def safe_mean(values):
    return sum(values) / len(values) if len(values) > 0 else 0.0


def log_train_scalars(writer, scalars, global_step):
    for name, value in scalars.items():
        writer.add_scalar(name, value, global_step=global_step)


class valid_module:
    def __init__(self, train_config):
        self.__init_test_data(train_config['validation']['testset'])
        self.__init_metric(train_config['validation']['iqas'])

        self.train_config = train_config

        if train_configs['generator_config']['loss_config']['perceptual']['model'] == 'vgg19':
            self.PERCEPTUAL_LOSS = loss.Perceptual_Loss(feature_layers=train_configs['generator_config']['loss_config']['perceptual']['layer'])

        self.ADVERSARIAL_LOSS = loss.D_LOSS()
        if train_configs['generator_config']['loss_config']['adversarial']['type'] == 'ragan':
            self.adv_loss_type = 'relative_opt_d'
        elif train_configs['generator_config']['loss_config']['adversarial']['type'] == 'gan':
            self.adv_loss_type = 'vanilla'

    def __init_test_data(self, valset_config):
        assert len(valset_config) > 0
        self.validset = []

        for i in range(len(valset_config)):
            validset_info = {}
            validset_info["name"] = valset_config[i]["name"]

            validset_info["lrs"] = functional.common.image_handle(file_io.reader.read_images(valset_config[i]["root"]+"/LR", ['.png', '.jpg', 'jepg']), normlize='zo', expand_dim=0, chw=True, to_tensor=False)
            validset_info["hrs"] = functional.common.image_handle(file_io.reader.read_images(valset_config[i]["root"]+"/HR", ['.png', '.jpg', 'jepg']), normlize='zo', expand_dim=0, chw=True, to_tensor=False)

            self.validset.append(validset_info)

    def __init_metric(self, metrices):
        assert len(metrices) > 0
        self.metrices = []

        for mn in metrices:
            metric = {}
            metric["name"] = mn
            if mn == "psnr":
                metric["cal"] = pyiqa.create_metric(mn, device=torch.device("cuda"), as_loss=False, test_y_channel=True, color_space='ycbcr')
            else:
                metric["cal"] = pyiqa.create_metric(mn, device=torch.device("cuda"), as_loss=False)

            self.metrices.append(metric)

    @torch.no_grad()
    def test(self, model, model_d):
        model.eval()
        results = []

        for valset in self.validset:
            srs = []

            for i in range(len(valset["lrs"])):
                    sr = model(torch.from_numpy(valset["lrs"][i]).cuda()).clone().detach()
                    srs.append(torch.clip(sr,0,1))

            temp = {}
            temp["dataset"] = valset["name"]
            for metric in self.metrices:
                value = 0
                for i in range(len(srs)):
                    value += float(metric["cal"](srs[i], torch.from_numpy(valset["hrs"][i]).cuda()))
                value /= len(srs)
                temp[metric["name"]] = value

            values = []
            for idx in range(len(srs)):
                values.append(float(self.PERCEPTUAL_LOSS(torch.from_numpy(valset["hrs"][idx]).cuda(), srs[idx], train_configs['generator_config']['loss_config']['perceptual']['weight'])))
            values = sum(values) / len(values)
            temp['perceptual_loss'] = values

            
            if self.train_config['validation']['need_gan_metric']:
                values1 = []
                values2 = []
                for idx in range(len(srs)):
                    d_sr = model_d(srs[idx].detach())
                    d_hr = model_d(torch.from_numpy(valset["hrs"][idx]).cuda())

                    if self.adv_loss_type == 'relative_opt_d':
                        values1.append(float(self.ADVERSARIAL_LOSS(d_sr, d_hr, type=self.adv_loss_type, weight=float(self.train_config['discriminator_config']['loss_weight']))))
                        values2.append(float(self.ADVERSARIAL_LOSS(d_sr, d_hr, type='relative_opt_g', weight=float(self.train_config['generator_config']['loss_config']['adversarial']['weight']))))
                    elif self.adv_loss_type == 'vanilla':
                        values1.append(float(self.ADVERSARIAL_LOSS(d_sr, torch.ones_like(d_sr), type=self.adv_loss_type, weight=float(self.train_config['discriminator_config']['loss_weight']))))
                        values2.append(float(self.ADVERSARIAL_LOSS(d_hr, torch.ones_like(d_hr), type=self.adv_loss_type, weight=float(self.train_config['discriminator_config']['loss_weight']))))
                values1 = sum(values1) / len(values1)
                values2 = sum(values2) / len(values2)
                if self.adv_loss_type == 'relative_opt_d' or\
                self.adv_loss_type == 'vanilla':
                    temp['d_adversarial_loss'] = values1
                    temp['g_adversarial_loss'] = values2

            results.append(temp)
        return results


if __name__ == '__main__':
    args = parse_args()
    logger.info(f'{datetime.now()} start train task')

    #--------------------------------------------------------------------------------------------
    # Phase-0:
    #   read config
    #   init models
    #--------------------------------------------------------------------------------------------
    train_configs = file_io.reader.read_yaml(args.config)
    train_configs['generator_config']['srf'] = train_configs['data_config']['srf']

    os.makedirs(train_configs['validation']['save_root'], exist_ok=True)
    file_io.base.copy_file(ori_file=args.config,
                           destination=train_configs['validation']['save_root'])

    #
    G, G_OPT, G_LR_SCHEDULE = torch_model.init_model(model_config=train_configs['generator_config'],
                                                     train_config=train_configs['train_config'])
    G.cuda()
    
    logger.info(f'generator init complete')

    if train_configs['discriminator_config']['discriminator_num'] == 1:
        Dvan, Dvan_OPT, Dvan_LR_SCHEDULE = torch_model.init_model(model_config=train_configs['discriminator_config'],
                                                                  train_config=train_configs['train_config'])
        Dvan.cuda()
    elif train_configs['discriminator_config']['discriminator_num'] == 2:
        # TODO
        ...

    logger.info(f'discriminator init complete')

    #--------------------------------------------------------------------------------------------
    # Phase-1:
    #   init dataloader   
    #   init tensorboard
    #   init validation module
    #--------------------------------------------------------------------------------------------
    _, data_prefecher = data_module.init_dataloader(train_configs['data_config'])
    writer = init_tensorboard(train_configs, args.config)
    logger.info(f'data prefecher init complete')

    VALIDATION = valid_module(train_configs)
    st = time.time()
    eval_results = VALIDATION.test(G, Dvan)
    log_validation(writer, eval_results, global_step=0)
    log_train_scalars(writer, {
        'train/loss/g_pixel': 0,
        'train/loss/g_perceptual': 0,
        'train/loss/g_adv': 0,
        'train/loss/d_adv': 0,
        'train/lr/generator': G_LR_SCHEDULE.get_last_lr()[-1],
        'train/lr/dvan': Dvan_LR_SCHEDULE.get_last_lr()[-1],
    }, global_step=0)

    
    et = time.time()
    logger.info(f'validation module init complete, exec-time{et-st}')

    #--------------------------------------------------------------------------------------------
    # Phase-2:
    #   init loss function
    #   init log writer
    #--------------------------------------------------------------------------------------------
    if train_configs['generator_config']['loss_config']['pixel_loss']['type'] == 'l1':
        PIXEL_LOSS = nn.L1Loss()
    
    if train_configs['generator_config']['loss_config']['perceptual']['model'] == 'vgg19':
        PERCEPTUAL_LOSS = loss.Perceptual_Loss(feature_layers=train_configs['generator_config']['loss_config']['perceptual']['layer'])

    ADVERSARIAL_LOSS = loss.D_LOSS()
    if train_configs['generator_config']['loss_config']['adversarial']['type'] == 'ragan':
        g_adv_type = 'relative_opt_g'
    elif train_configs['generator_config']['loss_config']['adversarial']['type'] == 'gan':
        g_adv_type = 'vanilla'

    if train_configs['discriminator_config']['loss_type'] == 'ragan':
        d_adv_type = 'relative_opt_d'
    elif train_configs['discriminator_config']['loss_type'] == 'gan':
        d_adv_type = 'vanilla'

    logger.info(f'loss module complete')

    #--------------------------------------------------------------------------------------------
    # Phase-3:
    #   SPR-Module init
    #   TODO: MS
    #--------------------------------------------------------------------------------------------
    hr_size = (train_configs['data_config']['hr_size'], 
               train_configs['data_config']['hr_size'])
    SPR_REC_1 = tensor_unit.Replacement_v2(r_size=train_configs['spr_config']['ranker_1'][0],
                                           size=hr_size, roll_prob=train_configs['spr_config']['roll_prob'])
    SPR_REC_2 = tensor_unit.Replacement_v2(r_size=train_configs['spr_config']['ranker_1'][1],
                                           size=hr_size, roll_prob=train_configs['spr_config']['roll_prob'])
    
    #--------------------------------------------------------------------------------------------
    # Phase-4:
    #   Train Phase
    #--------------------------------------------------------------------------------------------
    now_iter = 0
    now_epoch = 0
    bar = FillingSquaresBar("Train Epoch 1:", max=train_configs['train_config']['iter_per_epoch'])

    g_train_perceptual_loss = []
    g_train_pixel_loss      = []
    g_train_adv_loss        = []

    dvan_train_adv_loss     = []
    log_interval = int(train_configs.get('tensorboard', {}).get('log_interval', 100))

    assert train_configs['data_config']['batch_size'] % train_configs['generator_config']['accumulate_times'] == 0
    assert train_configs['data_config']['batch_size'] % train_configs['discriminator_config']['accumulate_times'] == 0

    while now_iter < train_configs['train_config']['train_iter']:
        data_prefecher.restart()
        epoch_start_time = time.time()

        if train_configs['data_config']['mode'] == 'realesrgan':
            batch_lrs, batch_hrs_usm, batch_hrs = data_prefecher.next()
        else:
            batch_lrs, batch_hrs = data_prefecher.next()

        while batch_lrs is not None and batch_hrs is not None:

            #------------------------------------------------------------------------------------
            # Train Discriminator
            #------------------------------------------------------------------------------------
            Dvan.train()
            Dvan_OPT.zero_grad()

            for d_accumulate_times in range(train_configs['discriminator_config']['accumulate_times']):

                act_stp = train_configs['data_config']['batch_size'] // train_configs['discriminator_config']['accumulate_times']

                act_lrs = batch_lrs[d_accumulate_times*act_stp : (d_accumulate_times+1)*act_stp]
                act_hrs = batch_hrs[d_accumulate_times*act_stp : (d_accumulate_times+1)*act_stp]

                act_srs = G(act_lrs)

                act_rec_1 = SPR_REC_1(HR=act_hrs, SR=act_srs,
                                      real_portion=train_configs['spr_config']['real_portion'],
                                      mode=train_configs['spr_config']['mode'], Dvan=Dvan, need_mask=False)
                
                act_rec_2 = SPR_REC_2(HR=act_hrs, SR=act_srs,
                                      real_portion=train_configs['spr_config']['real_portion'],
                                      mode=train_configs['spr_config']['mode'], Dvan=Dvan, need_mask=False)
                
                Dvan.train()

                dvan_srs   = Dvan(act_srs.detach())
                dvan_hrs   = Dvan(act_hrs.detach())
                dvan_rec_1 = Dvan(act_rec_1.detach())
                dvan_rec_2 = Dvan(act_rec_2.detach())

                if d_adv_type == 'relative_opt_d':
                    dvan_adversarial_loss = ADVERSARIAL_LOSS(dvan_srs, dvan_hrs, type=d_adv_type, weight=float(train_configs['discriminator_config']['loss_weight'])) +\
                                            ADVERSARIAL_LOSS(dvan_rec_1, dvan_srs, type=d_adv_type, weight=float(train_configs['discriminator_config']['loss_weight'])) +\
                                            ADVERSARIAL_LOSS(dvan_rec_2, dvan_rec_1, type=d_adv_type, weight=float(train_configs['discriminator_config']['loss_weight']))


                dvan_adversarial_loss /= train_configs['discriminator_config']['accumulate_times']

                dvan_train_adv_loss.append(float(dvan_adversarial_loss))

                dvan_adversarial_loss.backward()

            Dvan_OPT.step()
            Dvan_OPT.zero_grad()
            Dvan_LR_SCHEDULE.step()
            
            #------------------------------------------------------------------------------------
            #
            #------------------------------------------------------------------------------------
            G.train()
            G_OPT.zero_grad()

            functional.common.set_requires_grad(Dvan, False)

            for g_accumulate_times in range(train_configs['generator_config']['accumulate_times']):

                act_stp = train_configs['data_config']['batch_size'] // train_configs['generator_config']['accumulate_times']

                act_lrs = batch_lrs[g_accumulate_times*act_stp : (g_accumulate_times+1)*act_stp]
                act_hrs = batch_hrs[g_accumulate_times*act_stp : (g_accumulate_times+1)*act_stp]

                act_srs = G(act_lrs)

                dvan_srs = Dvan(act_srs)

                act_rec_1 = SPR_REC_1(HR=act_hrs, SR=act_srs,
                                      real_portion=train_configs['spr_config']['real_portion'],
                                      mode=train_configs['spr_config']['mode'], Dvan=Dvan, need_mask=False)
                
                act_rec_2 = SPR_REC_2(HR=act_hrs, SR=act_srs,
                                      real_portion=train_configs['spr_config']['real_portion'],
                                      mode=train_configs['spr_config']['mode'], Dvan=Dvan, need_mask=False)


                g_pixel_loss       = PIXEL_LOSS(act_srs, act_hrs) * train_configs['generator_config']['loss_config']['pixel_loss']['weight']
                g_perceptual_loss  = PERCEPTUAL_LOSS(act_hrs, act_srs, train_configs['generator_config']['loss_config']['perceptual']['weight']) +\
                                     PERCEPTUAL_LOSS(act_hrs, act_rec_1, train_configs['generator_config']['loss_config']['perceptual']['weight']) +\
                                     PERCEPTUAL_LOSS(act_hrs, act_rec_2, train_configs['generator_config']['loss_config']['perceptual']['weight'])
                
                if g_adv_type == 'relative_opt_g':
                    dvan_hrs = Dvan(act_hrs)
                    dvan_rec_1 = Dvan(act_rec_1)
                    dvan_rec_2 = Dvan(act_rec_2)
                    g_adversarial_loss = ADVERSARIAL_LOSS(dvan_srs, dvan_hrs, type=g_adv_type, weight=float(train_configs['generator_config']['loss_config']['adversarial']['weight'])) +\
                                         ADVERSARIAL_LOSS(dvan_rec_1, dvan_hrs, type=g_adv_type, weight=float(train_configs['generator_config']['loss_config']['adversarial']['weight'])) +\
                                         ADVERSARIAL_LOSS(dvan_rec_2, dvan_hrs, type=g_adv_type, weight=float(train_configs['generator_config']['loss_config']['adversarial']['weight']))

                g_train_pixel_loss.append(float(g_pixel_loss))
                g_train_perceptual_loss.append(float(g_perceptual_loss))
                g_train_adv_loss.append(float(g_adversarial_loss))

                g_loss = g_pixel_loss + g_perceptual_loss + g_adversarial_loss
                g_loss /= train_configs['generator_config']['accumulate_times']

                g_loss.backward()
            
            
            G_OPT.step()
            G_OPT.zero_grad()
            G_LR_SCHEDULE.step()

            functional.common.set_requires_grad(Dvan, True)

            #------------------------------------------------------------------------------------
            #
            #------------------------------------------------------------------------------------
            now_iter += 1
            bar.next()

            if log_interval > 0 and now_iter % log_interval == 0:
                log_train_scalars(writer, {
                    'train/loss_iter/g_pixel': safe_mean(g_train_pixel_loss[-train_configs['generator_config']['accumulate_times']:]),
                    'train/loss_iter/g_perceptual': safe_mean(g_train_perceptual_loss[-train_configs['generator_config']['accumulate_times']:]),
                    'train/loss_iter/g_adv': safe_mean(g_train_adv_loss[-train_configs['generator_config']['accumulate_times']:]),
                    'train/loss_iter/d_adv': safe_mean(dvan_train_adv_loss[-train_configs['discriminator_config']['accumulate_times']:]),
                    'train/lr/generator': G_LR_SCHEDULE.get_last_lr()[-1],
                    'train/lr/dvan': Dvan_LR_SCHEDULE.get_last_lr()[-1],
                }, global_step=now_iter)

            if train_configs['data_config']['mode'] == 'realesrgan':
                batch_lrs, batch_hrs_usm, batch_hrs = data_prefecher.next()
            else:
                batch_lrs, batch_hrs = data_prefecher.next()

            if now_iter % train_configs['train_config']['iter_per_epoch'] == 0:
                bar.finish()
                now_epoch += 1
                bar = FillingSquaresBar(f"Train Epoch {now_epoch+1}:", max=train_configs['train_config']['iter_per_epoch'])

                path = os.path.join(train_configs['validation']['save_root'], 'models', str(now_epoch))
                os.makedirs(path)

                torch.save(G.state_dict(), os.path.join(path, 'G.pth'))
                torch.save(Dvan.state_dict(), os.path.join(path, 'Dvan.pth'))

                eval_results = VALIDATION.test(G.eval(), Dvan.eval())
                log_validation(writer, eval_results, global_step=now_iter)
                log_train_scalars(writer, {
                    'train/loss/g_pixel': safe_mean(g_train_pixel_loss),
                    'train/loss/g_perceptual': safe_mean(g_train_perceptual_loss),
                    'train/loss/g_adv': safe_mean(g_train_adv_loss),
                    'train/loss/d_adv': safe_mean(dvan_train_adv_loss),
                    'train/lr/generator': G_LR_SCHEDULE.get_last_lr()[-1],
                    'train/lr/dvan': Dvan_LR_SCHEDULE.get_last_lr()[-1],
                    'train/time/epoch_seconds': time.time() - epoch_start_time,
                    'train/progress/epoch': now_epoch,
                }, global_step=now_iter)
                writer.flush()

                g_train_perceptual_loss = []
                g_train_pixel_loss      = []
                g_train_adv_loss        = []

                dvan_train_adv_loss     = []
                
                print(f'Epoch time: {time.time()-epoch_start_time}')
                epoch_start_time = time.time()
            ...
        

    
