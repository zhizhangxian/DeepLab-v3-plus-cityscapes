#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import random
import numpy as np

class Config(object):
    def __init__(self, multi_scale=False):
        ## model and loss
        self.ignore_label = 255
        self.aspp_global_feature = False
        # dataset
        self.n_classes = 19
        self.datapth = '/seu_share/home/wkyang/datasets/cityscapes'
        self.n_workers = 4
        self.crop_size = (768, 768)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        # optimizer
        self.warmup_steps = 1000
        self.warmup_start_lr = 5e-6
        self.lr_start = 1e-2
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.lr_power = 0.9
        self.max_iter = 41000
        # training control
        self.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        self.flip = True
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.ims_per_gpu = 4
        self.msg_iter = 100
        self.eval_iter = 10
        self.ohem_thresh = 0.7
        self.respth = './res'
        self.port = 32168
        # eval control
        self.seed_max = 1000
        self.eval_batchsize = 2
        self.eval_n_workers = 2
        self.eval_scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75) if multi_scale else (1.0,)
        self.eval_flip = True
        self.alpha = 0.25
        self.beta = 0.25

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)