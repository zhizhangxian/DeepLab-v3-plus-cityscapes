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


model = Deeplab_v3plus(cfg, args, alphas=ops_list, betas=net_arch, gammas=skip_feature_list)

x = torch.randn(2, 3, 128, 128)

print(model(x).shape)


params, flops = profile(model, inputs=(x,))
print(params)
print(flops)
