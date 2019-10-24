#!/usr/bin/python
# -*- encoding: utf-8 -*-
import platform

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import Resnet101

if platform.system() is 'Windows':
    from torch.nn import BatchNorm2d
else:
    from modules import InPlaceABNSync as BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=True)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Deeplab_v3plus(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(Deeplab_v3plus, self).__init__()
        self.backbone = Resnet101(stride=16)
        self.decoder = nn.Conv2d(in_channels=2048, out_channels=19, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        _, _, _, feat = self.backbone(x)
        logits = self.decoder(feat)
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)

        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        back_bn_params, back_no_bn_params = self.backbone.get_params()
        # tune_wd_params = list(self.aspp.parameters()) \
        #     + list(self.decoder.parameters()) \
        #     + back_no_bn_params
        tune_wd_params = list(self.decoder.parameters()) + back_no_bn_params

        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params


if __name__ == "__main__":
    from configs.configurations import Config

    net = Deeplab_v3plus(Config())
    net.cuda()
    with torch.no_grad():
        net = nn.DataParallel(net)
        for i in range(1):
            #  with torch.no_grad():
            in_ten = torch.randn((2, 3, 768, 768)).cuda()
            logits = net(in_ten)
            print(i)
            print(logits.size())
