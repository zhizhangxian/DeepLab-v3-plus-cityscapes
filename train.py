#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import *
from models.deeplabv3plus import Deeplab_v3plus
from cityscapes import CityScapes, collate_fn2
from evaluate import MscEval
from optimizer import Optimizer
from loss import OhemCELoss, pgc_loss
from configs import config_factory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import logging
import time
import datetime
import argparse


cfg = config_factory['resnet_cityscapes']
if not osp.exists(cfg.respth): os.makedirs(cfg.respth)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()


def train(verbose=True, **kwargs):
    args = kwargs['args']
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
                backend = 'nccl',
                init_method = 'tcp://127.0.0.1:{}'.format(cfg.port),
                world_size = torch.cuda.device_count(),
                rank = args.local_rank
                )
    setup_logger(cfg.respth)
    logger = logging.getLogger()

    ## dataset
    ds = CityScapes(cfg, mode='train', num_copys=2)
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size = cfg.ims_per_gpu,
                    shuffle = False,
                    sampler = sampler,
                    num_workers = cfg.n_workers,
                    collate_fn=collate_fn2,
                    pin_memory = True,
                    drop_last = True)

    ## model
    net = Deeplab_v3plus(cfg)
    net.train()
    net.cuda()
    net = nn.parallel.DistributedDataParallel(net,
            device_ids = [args.local_rank, ],
            output_device = args.local_rank
            )
    n_min = cfg.ims_per_gpu*cfg.crop_size[0]*cfg.crop_size[1]//16
    criteria = OhemCELoss(thresh=cfg.ohem_thresh, n_min=n_min).cuda()
    Criterion = pgc_loss(use_pgc = [0,1,2], criteria=criteria)
    ## optimizer
    optim = Optimizer(
            net,
            cfg.lr_start,
            cfg.momentum,
            cfg.weight_decay,
            cfg.warmup_steps,
            cfg.warmup_start_lr,
            cfg.max_iter,
            cfg.lr_power
            )
    alpha, beta = cfg.alpha, cfg.beta
    ## train loop
    loss_avg = []
    pgc_avg = []
    ce_avg = []
    ssp_avg = []
    ohem_avg = []

    st = glob_st = time.time()
    diter = iter(dl)
    n_epoch = 0
    for it in range(cfg.max_iter):
        try:
            im, lb, overlap, flip = next(diter)
            if not im.size()[0]!=cfg.ims_per_gpu // 2:
                continue
        except StopIteration:
            n_epoch += 1
            sampler.set_epoch(n_epoch)
            diter = iter(dl)
            im, lb, overlap, flip = next(diter)
        im = im.cuda()
        lb = lb.cuda()

        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)
        optim.zero_grad()
        im1, im2 = im[::2], im[1::2]
        lb1, lb2 = lb[::2], lb[1::2]
        logits1 = net(im1)
        logits2 = net(im2)
        # logits = torch.cat([logits1[-1], logits2[-1]], dim=0)

        outputs = []
        for f1, f2 in zip(logits1, logits2):
            outputs.append([f1, f2])
        logits = torch.cat([logits1[-1], logits2[-1]], dim=0)

        mse, sym_ce, mid_mse, mid_ce, mid_l1, ce = Criterion(outputs, overlap, flip, lb)
        # loss = criteria(logits, lb)
        loss = beta * sym_ce + ce
        gc_loss = sum(mid_mse)
        loss += alpha * gc_loss
        loss.backward()

        optim.step()

        loss_avg.append(loss.item())
        ohem_avg.append(ce.item())
        pgc_avg.append(gc_loss.item())
        ssp_avg.append(sym_ce.item())
        ## print training log message
        if it%cfg.msg_iter==0 and not it==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            ohem = sum(ohem_avg) / len(ohem_avg) 
            pgc = sum(pgc_avg) / len(pgc_avg) 
            ssp = sum(ssp_avg) / len(ssp_avg) 
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((cfg.max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds = eta))
            msg = ', '.join([
                    'iter: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'ohem: {ohem:.4f}', 
                    'pgc: {pgc:.4f}', 
                    'ssp: {ssp:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it,
                    max_it = cfg.max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta,
                    ohem = ohem,
                    pgc = pgc,
                    ssp = ssp,
                )
            logger.info(msg)
            loss_avg = []
            pgc_avg = []
            ssp_avg = []
            ohem_avg = []
            st = ed

    ## dump the final model and evaluate the result
    if verbose:
        net.cpu()
        save_pth = osp.join(cfg.respth, 'model_final.pth')
        state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        if dist.get_rank()==0: torch.save(state, save_pth)
        logger.info('training done, model saved to: {}'.format(save_pth))
        logger.info('evaluating the final model')
        net.cuda()
        net.eval()
        evaluator = MscEval(cfg)
        mIOU = evaluator(net)
        logger.info('mIOU is: {}'.format(mIOU))


if __name__ == "__main__":
    args = parse_args()
    train(args=args)
