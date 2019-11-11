import platform
import torch.nn as nn

if platform.system() == 'Windows':
    from torch.nn import BatchNorm2d as BatchNorm2d
else:
    print("=> warning! using ABN")
    from modules import InPlaceABNSync as BatchNorm2d
    
OPS = {
    'identity': lambda C_in, C_out, stride, affine: Identity(C_in, C_out, stride, affine),
    'conv3x3': lambda C_in, C_out, stride, affine: ConvBNReLU(C_in, C_out, 3, stride, 1, 1, affine=affine),
    'conv5x5': lambda C_in, C_out, stride, affine: ConvBNReLU(C_in, C_out, 5, stride, 2, 1, affine=affine),
    'conv3x3x2': lambda C_in, C_out, stride, affine: nn.Sequential(ConvBNReLU(C_in, C_out, 3, stride, 1, 1, affine=affine), ConvBNReLU(C_out, C_out, 3, stride, 1, 1, affine=affine)),
    'dil_conv_3x3': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, 2, affine=affine, use_ABN=True),
    'dil_conv_5x5': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 5, stride, 4, 2, affine=affine, use_ABN=True), 
    'bottle_3x3': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, True, True),
    'bottle_5x5': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, True, True),
}



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


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, use_ABN=True):
        super(DilConv, self).__init__()
        if not use_ABN:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                          padding=padding, groups=C_in, dilation=dilation, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )
        else:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                          padding=padding, groups=C_in, dilation=dilation, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                BatchNorm2d(C_out),
            )

    def forward(self, x):
        return self.op(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, use_ABN=True):
        super(SepConv, self).__init__()
        if use_ABN:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                          bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                BatchNorm2d(C_in, affine=affine),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                BatchNorm2d(C_out, affine=affine)
            )

        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in,
                          bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_in, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x):
        return self.op(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class HalfConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, use_ABN=True):
        super(HalfConv, self).__init__()
        if not use_ABN:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=(kernel_size, 1),
                          stride=stride, padding=(padding, 0), groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size),
                          stride=stride, padding=(0, padding), groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_in, affine=affine),
                nn.ReLU(inplace=False),
            )
        else:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=(kernel_size, 1),
                          stride=stride, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_in, kernel_size=(1, kernel_size),
                          stride=stride, padding=padding, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
                BatchNorm2d(C_out),
            )

    def forward(self, x):
        return self.op(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Identity(nn.Module):

    def __init__(self, C_in, C_out, stride, affine):
        super(Identity, self).__init__()
        self.op = ConvBNReLU(C_in, C_out, 1, stride, 0, 1, affine=affine) if C_in != C_out else None

    def forward(self, x):
        return self.op(x) if self.op is not None else x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)
