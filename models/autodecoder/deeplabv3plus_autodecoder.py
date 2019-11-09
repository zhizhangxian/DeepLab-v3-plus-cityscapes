import torch
import torch.nn.functional as F

from models.autodecoder.operations_decoder import *
from models.autodecoder.genotypes_decoder import PRIMITIVES


class retrain_cell(nn.Module):
    def __init__(self, C_in, C_out, C_skip, C_low, upsample, ops):
        super(retrain_cell, self).__init__()
        self._upsample = upsample
        if C_skip != -1 and C_low != -1:
            self._pre_skip_process = ConvBNReLU(C_skip, C_low, 1, 1, 0, affine=True)
            self._op = OPS[PRIMITIVES[ops]](C_in + C_low, C_out, 1, True)
        else:
            self._op = OPS[PRIMITIVES[ops]](C_in, C_out, 1, True)

    @staticmethod
    def scale_dimension(dim, scale):
        assert isinstance(dim, int)
        return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)

    def prev_feature_resize(self, prev_feature, mode):

        if mode == 'down':

            feature_size_h = self.scale_dimension(prev_feature.shape[2], 0.5)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 0.5)

        elif mode == 'up':

            feature_size_h = self.scale_dimension(prev_feature.shape[2], 2)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 2)

        else:
            raise NotImplementedError

        return F.interpolate(prev_feature, (feature_size_h, feature_size_w), mode='bilinear', align_corners=True)

    def forward(self, x, skip_feature):

        if self._upsample:
            x = self.prev_feature_resize(prev_feature=x, mode='up')

        if skip_feature is not None:

            if self._pre_skip_process is not None:
                skip_feature = self._pre_skip_process(skip_feature)

            if skip_feature.shape[2:] != x.shape[2:]:
                skip_feature = F.interpolate(skip_feature, x.shape[2:], mode='bilinear', align_corners=True)

            return(self._op(torch.cat((x, skip_feature), dim=1)))

        return self._op(x)


class AutoDecoder(nn.Module):
    def __init__(self, num_classes, args, alphas, betas, gammas, C_low=None):

        super(AutoDecoder, self).__init__()
        self._args = args
        self.net_arch = betas
        self.ops_list = alphas
        self.skip_feature_list = gammas
        self.cells = nn.ModuleList()
        self._num_classes = num_classes
        self._num_layers = args.decoder_layer
        assert self._num_layers >= 6, ValueError("number of decoder {:} is too small, you must set np less than 6!".format(self._num_layers))
        self.C_low = 48 if C_low is None else C_low
        self._block_multiplier = args.block_multiplier
        self._filter_multiplier = args.filter_multiplier
        self._low_feature_list, C_encoder_out = self._build_encoder(args)

        self.filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        self.decoder_cells = self._get_decoder(self.ops_list, self.net_arch, self.skip_feature_list, args)

        # self.C_last = self._filter_multiplier * self._block_multiplier * self.filter_param_dict[self.net_arch[-1]]
        self.output_conv = nn.Conv2d(256, self._num_classes, 1, bias=False)

    @staticmethod
    def _build_encoder(args):

        if args.backbone == 'resnet':

            low_feature_list = [256, 512, 1024, 2048]
            C_encoder_out = 2048
            return low_feature_list, C_encoder_out

        else:
            raise NotImplementedError

    def _get_decoder(self, ops_list, net_arch, skip_feature_list, args):

        prev_arch = net_arch[0]
        _cells = nn.ModuleList()
        C_in = args.block_multiplier * args.filter_multiplier * self.filter_param_dict[net_arch[0]]

        for i, arch in enumerate(net_arch):
            # C_in = args.block_multiplier * args.filter_multiplier * self.filter_param_dict[net_arch[i - 1]] if i else C_in

            if skip_feature_list[i] != -1:

                C_skip = self._low_feature_list[skip_feature_list[i]]
                if arch != prev_arch:
                    C_out = C_in
                    cell = retrain_cell(C_in, C_out, C_skip, self.C_low, upsample=1, ops=ops_list[i])
                    # C_in = C_out
                else:
                    cell = retrain_cell(C_in, C_in, C_skip, self.C_low, upsample=0, ops=ops_list[i])
            else:
                if arch != prev_arch:
                    C_out = C_in 
                    cell = retrain_cell(C_in, C_out, -1, -1, upsample=1, ops=ops_list[i])
                    # C_in = C_out
                else:
                    cell = retrain_cell(C_in, C_in, -1, -1, upsample=0, ops=ops_list[i])
            prev_arch = arch
            _cells.append(cell)
        return _cells

    def forward(self, x, low_level_feature_list):

        for i in range(self._num_layers):
            
            if self.skip_feature_list[i] != -1:
                skip_feature = low_level_feature_list[self.skip_feature_list[i]]
            else:
                skip_feature = None

            x = self.decoder_cells[i](x, skip_feature)
        return self.output_conv(x)
