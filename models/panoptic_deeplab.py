#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
import torch.nn.functional as F
import torchvision

from models.resnet import Resnet101
# from modules import InPlaceABNSync as BatchNorm2d
from torch.nn import BatchNorm2d


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
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1)
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


class Decoder(nn.Module):
    def __init__(self, n_classes, low_chan_4=256, low_chan_8=512, *args, **kwargs):
        super(Decoder, self).__init__()
        self.low_conv_feat4 = ConvBNReLU(low_chan_4, 32, 1, 1, 0)
        self.low_conv_feat8 = ConvBNReLU(low_chan_8, 64, 1, 1, 0)
        self.Conv_8_4 = ConvBNReLU(320, 256, 5, 1, 2)
        self.Conv_4 = ConvBNReLU(288, 256, 5, 1, 2)
        self.output_conv = nn.Sequential(ConvBNReLU(256, 256, 5, 1, 2),
                                         nn.Conv2d(256, n_classes, 1, bias=False))

    def forward(self, feat_aspp, feat_4, feat_8):
        feat_4 = self.low_conv_feat4(feat_4)
        feat_8 = self.low_conv_feat8(feat_8)
        feat_aspp_8 = F.interpolate(feat_aspp, feat_8.shape[2:], mode='bilinear', align_corners=True)
        feat_aspp_4 = F.interpolate(self.Conv_8_4(torch.cat((feat_aspp_8, feat_8), dim=1)), feat_4.shape[2:], mode='bilinear', align_corners=True)
        feat_out = self.Conv_4(torch.cat((feat_aspp_4, feat_4), dim=1))



        return self.output_conv(feat_out)

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
        self.aspp = ASPP(in_chan=2048, out_chan=256, with_gp=cfg.aspp_global_feature)
        self.decoder = Decoder(cfg.n_classes, low_chan=256)
        #  self.backbone = Darknet53(stride=16)
        #  self.aspp = ASPP(in_chan=1024, out_chan=256, with_gp=False)
        #  self.decoder = Decoder(cfg.n_classes, low_chan=128)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, feat8, _, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        logits = self.decoder(feat_aspp, feat4, feat8)

        print(logits.shape)

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
    from configs.configurations import Config

    cfg = Config()
    net = Deeplab_v3plus(cfg)
    net.cuda()
    net.train()
    net = nn.DataParallel(net)
    for i in range(100):
        #  with torch.no_grad():
        in_ten = torch.randn((1, 3, 768, 768)).cuda()
        logits = net(in_ten)
        print(i)
        print(logits.size())
