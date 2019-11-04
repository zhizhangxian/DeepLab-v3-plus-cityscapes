from configs import Config
from models.deeplabv3plus import Deeplab_v3plus as V3
from thop import profile
from models.autodecoder.deeplab_deocder_distributed import Deeplab_v3plus
import torch
from configs.basic_args import obtain_search_args
from configs.configurations import Config

import numpy as np

cfg = Config()
args = obtain_search_args()
net_arch = np.load('result/net_arch.npy')
ops_list = np.load('result/ops_list.npy')
skip_feature_list = np.load('result/skip_feature_list.npy')

print(net_arch)
print(ops_list)
print(skip_feature_list)

ops_list = [0, 0, 0, 0, 3, 0]
net_arch = np.array([3, 3, 3, 2, 1, 1])
skip_feature_list = np.array([-1, -1, -1, -1, 0, -1])

model = Deeplab_v3plus(cfg, args, alphas=ops_list, betas=net_arch, gammas=skip_feature_list)


cfg = Config()
model1 = V3(cfg)


print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.decoder.parameters()) / 1000000.0))


x = torch.randn(2, 3, 128, 128)
print(model(x).shape)


feat4, feat8, feat16, feat32 = model.backbone(x)

params, flops = profile(model, inputs=(x,))
print(params)
print(flops)


params, flops = profile(model1, inputs=(x,))
print(params)
print(flops)