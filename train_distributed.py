import os
import time
import logging
import datetime
import os.path as osp


import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader

from loss import OhemCELoss
from optimizer import Optimizer
from logger import setup_logger
from cityscapes import CityScapes
from configs import Config, set_seed
from models.deeplabv3plus import Deeplab_v3plus


def setup(rank, world_size, cfg):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = cfg.port

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    set_seed(cfg.seed)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, cfg):
    setup(rank, world_size, cfg)
    torch.cuda.device(rank)
    setup_logger(cfg.respth, rank)
    logger = logging.getLogger()

    train_dataset = CityScapes(cfg, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(train_dataset, batch_size=cfg.ims_per_gpu, shuffle=False, sampler=sampler, num_workers=cfg.n_workers, pin_memory=True, drop_last=True)

    net = Deeplab_v3plus(cfg)
    net.train()
    net.cuda(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=rank)

    n_min = cfg.ims_per_gpu*cfg.crop_size[0]*cfg.crop_size[1]//16
    criterion = OhemCELoss(thresh=cfg.ohem_thresh, n_min=n_min).cuda(rank)

    optim = Optimizer(net, cfg.lr_start, cfg.momentum, cfg.weight_decay, cfg.warmup_steps, cfg.warmup_start_lr, cfg.max_iter, cfg.lr_power)

    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dataloader)
    n_epoch = 0

    for it in range(cfg.max_iter):
        try:
            im, lb = next(diter)
            if not im.shape[0] == cfg.ims_per_gpu:
                continue

        except StopIteration:
            n_epoch += 1
            sampler.set_epoch(n_epoch)
            diter = iter(dataloader)
            im, lb = next(diter)
        im, lb = im.cuda(), lb.cuda()
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        logits = net(im)
        loss = criterion(logits, lb)
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        if it % cfg.msg_iter == 0 and not it == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed-st, ed - glob_st
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
            st - ed

    net.cpu()
    save_pth = osp.join(cfg.respth, 'model_final.pth')
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))

    cleanup()


def main():
    cfg = Config()
    if not osp.exists(cfg.respth):
        os.mkdir(cfg.respth)
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, cfg), nprocs=world_size)


if __name__ == "__main__":
    main()
