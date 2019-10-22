#!/usr/bin/python
# -*- encoding: utf-8 -*-
import platform

import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
import torch.nn.functional as F
import torchvision

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


class Decoder(nn.Module):
    def __init__(self, n_classes, low_chan=256, *args, **kwargs):
        super(Decoder, self).__init__()
        self.conv_low = ConvBNReLU(low_chan, 48, ks=1, padding=0)
        self.conv_cat = nn.Sequential(
            ConvBNReLU(304, 256, ks=3, padding=1),
            ConvBNReLU(256, 256, ks=3, padding=1),
        )
        self.conv_out = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, feat_low, feat_aspp):
        H, W = feat_low.size()[2:]
        feat_low = self.conv_low(feat_low)
        feat_aspp_up = F.interpolate(feat_aspp, (H, W), mode='bilinear',
                                     align_corners=True)
        feat_cat = torch.cat([feat_low, feat_aspp_up], dim=1)
        feat_out = self.conv_cat(feat_cat)
        logits = self.conv_out(feat_out)
        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class MacroDecoder(nn.Module):
    def __init__(self, C_low_level_feature_list):
        super(MacroDecoder, self).__init__()
        self.cell_0 = ConvBNReLU(256 + 192, 256, 5, 1, 1)  # level_16
        self.cell_1 = ConvBNReLU(256 + 96, 256, 5, 1, 1)  # level_8
        self.cell_2 = ConvBNReLU(256 + 48, 256, 5, 1, 1)  # level_4

        self.skip_conv1 = ConvBNReLU(C_low_level_feature_list[0], 48, 1, 1, 0)
        self.skip_conv2 = ConvBNReLU(C_low_level_feature_list[1], 96, 1, 1, 0)
        self.skip_conv3 = ConvBNReLU(C_low_level_feature_list[2], 192, 1, 1, 0)

        self.output_conv = ConvBNReLU(256, 19, 1, 1, 0)

    def forward(self, x, feature_4, feature_8, feature_16):
        feature_4 = self.skip_conv1(feature_4)
        feature_8 = self.skip_conv2(feature_8)
        feature_16 = self.skip_conv3(feature_16)

        x = self.cell_0(torch.cat((x, feature_16), dim=1))
        x = self.cell_1(torch.cat((F.interpolate(x, feature_8.shape[2:], mode='bilinear', align_corners=True), feature_8), dim=1))
        x = self.cell_2(torch.cat((F.interpolate(x, feature_4.shape[2:], mode='bilinear', align_corners=True), feature_4), dim=1))

        return self.output_conv(x)

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
        low_level_list = [256, 512, 1024]
        self.decoder = MacroDecoder(low_level_list)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)

        logits = self.decoder(feat_aspp, feat4, feat8, feat16)
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
