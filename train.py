#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import *
from models.deeplabv3plus import Deeplab_v3plus
from cityscapes import CityScapes
from evaluate import MscEval
from optimizer import Optimizer
from loss import OhemCELoss
from configs import config_factory
from configs import set_seed

import os
import logging
import random
import time
import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

cfg = config_factory['resnet_cityscapes']
if not osp.exists(cfg.respth):
    os.makedirs(cfg.respth)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1)
    parse.add_argument('--resume', default=None, type=str)
    return parse.parse_args()


def train(verbose=True, **kwargs):
    args = kwargs['args']
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(cfg.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    setup_logger(cfg.respth)
    logger = logging.getLogger()
    seed = random.randint(1, cfg.seed_max)
    set_seed(seed)

    if dist.get_rank() == 0:
        msg = 'random seed: {:}'.format(seed)
        logger.info(msg)

    # dataset
    ds = CityScapes(cfg, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size=cfg.ims_per_gpu,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=cfg.n_workers,
                    pin_memory=True,
                    drop_last=True)

    # model
    net = Deeplab_v3plus(cfg)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])

    net.train()
    net.cuda()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[args.local_rank, ],
                                              output_device=args.local_rank
                                              )
    n_min = cfg.ims_per_gpu*cfg.crop_size[0]*cfg.crop_size[1]//16
    criteria = OhemCELoss(thresh=cfg.ohem_thresh, n_min=n_min).cuda()

    # optimizer
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

    # train loop
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)

    try:
        if args.resume:
            n_epoch = checkpoint['n_epoch']
            start_iter = checkpoint['it']
            optim.optim.load_state_dict(checkpoint['optimizer'])
        else:
            n_epoch = 0
            start_iter = 0
    except:
        raise NotImplementedError

    for it in range(cfg.max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == cfg.ims_per_gpu:
                continue
        except StopIteration:
            n_epoch += 1
            sampler.set_epoch(n_epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        logits = net(im)
        loss = criteria(logits, lb)
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        # print training log message
        if it % cfg.msg_iter == 0 and not it == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((cfg.max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                'iter: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=it,
                max_it=cfg.max_iter,
                lr=lr,
                loss=loss_avg,
                time=t_intv,
                eta=eta
            )
            logger.info(msg)
            loss_avg = []
            st = ed

        if dist.get_rank() == 0:
            if it < (cfg.max_iter - 20 * cfg.msg_iter):
                if it % int(20 * cfg.msg_iter) == 0:
                    save_pth = osp.join(cfg.respth, 'iter_{:}_'.format(it) + 'model.pth')
                    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                    torch.save({'it': it, 'state_dict': state, 'optimizer': optim.optim.state_dict(), 'n_epoch': n_epoch}, save_pth)

            else:
                if it % int(cfg.msg_iter) == 0:
                    save_pth = osp.join(cfg.respth, 'iter_{:}_'.format(it) + 'model.pth')
                    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                    torch.save({'it': it, 'state_dict': state, 'optimizer': optim.optim.state_dict(), 'n_epoch': n_epoch}, save_pth)

    # dump the final model and evaluate the result
    if verbose:
        net.cpu()
        save_pth = osp.join(cfg.respth, 'model_final.pth')
        state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        if dist.get_rank() == 0:
            torch.save(state, save_pth)
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
