#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *



class CityScapes(Dataset):
    def __init__(self, cfg, mode='train', num_copys=1, *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.cfg = cfg

        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(cfg.datapth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(cfg.datapth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
            ])
        ## pre-processing
        if num_copys == 1:

            self.trans = Compose([
                ColorJitter(
                    brightness = cfg.brightness,
                    contrast = cfg.contrast,
                    saturation = cfg.saturation),
                RandomScale(cfg.scales),
                RandomCrop(cfg.crop_size),
                HorizontalFlip(),

                ])
        elif num_copys == 2:
            self.to_tensor = Pair_ToTensor(self.to_tensor)
            img_size = (1024,2048)
            self.trans = Compose([
                Pair_RandomScale(cfg.scales, img_size),
                Pair_ColorJitter(
                    brightness = cfg.brightness,
                    contrast = cfg.contrast,
                    saturation = cfg.saturation),
                Pair_RandomCrop(cfg.crop_size),
                Pair_HorizontalFlip(),
                ])

        self.num_copys = num_copys

    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans(im_lb)
            img, label = im_lb['im'], im_lb['lb']

        if self.num_copys == 1:
            imgs = self.to_tensor(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]
            label = self.convert_labels(label)
            return imgs, label

        else:
            im_lb = self.to_tensor(im_lb)
            return im_lb
            


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label


def collate_fn2(batchs):
    _imgs = []
    _targets = []
    _overlaps = []
    flips = []
    for index, batch in enumerate(batchs):
        _overlaps.append([])
        imgs, targets, overlaps, flip = batch['im'], batch['lb'], batch['overlap'], batch['flip']
        
        _flip = 1
        for i in range(len(imgs)):
            # print('index: {:}, i: {:}'.format(index, i))
            _imgs.append(torch.unsqueeze(imgs[i], 0))
            _targets.append(torch.unsqueeze(targets[i], 0))
            _overlaps[index].append(overlaps[i])
            _flip *= flip[i]
        flips.append(_flip)
    return torch.cat(_imgs, dim=0), torch.cat(_targets, dim=0), _overlaps, flips





if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import os
    os.chdir('../')
    from configs.configurations import Config

    cfg = Config()
    ds = CityScapes(cfg, mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
