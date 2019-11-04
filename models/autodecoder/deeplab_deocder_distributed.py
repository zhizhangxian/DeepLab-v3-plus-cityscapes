#!/usr/bin/python
# -*- encoding: utf-8 -*-

from models.autodecoder.deeplabv3plus_autodecoder import AutoDecoder
import torch
import platform
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import Resnet101

if platform.system() is 'Windows':
    from torch.nn import BatchNorm2d as BatchNorm2d
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


class ASPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, with_gp=True, *args, **kwargs):
        super(ASPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=6, padding=6)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=12, padding=12)
        self.conv4 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=18, padding=18)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1, padding=0)
            self.conv_out = ConvBNReLU(out_chan * 5, out_chan, ks=1, padding=0)
        else:
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan, ks=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Deeplab_v3plus(nn.Module):
    def __init__(self, cfg, args, alphas, betas, gammas, **kwargs):
        super(Deeplab_v3plus, self).__init__()
        self.backbone = Resnet101(stride=16)
        self.filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        # self.C_aspp_out = args.filter_multiplier * args.block_multiplier * self.filter_param_dict[betas[0]]
        self.C_aspp_out = args.filter_multiplier * 8
        self.aspp = ASPP(in_chan=2048, out_chan=self.C_aspp_out, with_gp=cfg.aspp_global_feature)
        self.decoder = AutoDecoder(19, args, alphas=alphas, betas=betas, gammas=gammas)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feature_4, feature_8, feature_16, feature_32 = self.backbone(x)
        x = self.aspp(feature_32)
        skip_feature_list = [feature_4, feature_8, feature_16, feature_32]
        logits = self.decoder(x, skip_feature_list)
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
        tune_wd_params = list(self.aspp.parameters()) \
            + list(self.decoder.parameters()) \
            + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params


if __name__ == "__main__":
    net = Deeplab_v3plus(19)
    net.cuda()
    net.train()
    net = nn.DataParallel(net)
    for i in range(100):
        #  with torch.no_grad():
        in_ten = torch.randn((1, 3, 768, 768)).cuda()
        logits = net(in_ten)
