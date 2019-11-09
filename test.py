from configs.configurations import Config
from configs.basic_args import obtain_search_args
from models.autodecoder.deeplab_deocder import Deeplab_v3plus

# from models.deeplabv3plus import Deeplab_v3plus



# from models.deeplabv3plus import Deeplab_v3plus as Deeplab_v3plus_naive
import torch

arch = torch.load(r'H:\November\7.nov\result_decoder\new_Standard_result.pth.tar')

print(arch)

net_arch = arch['net_arch']
ops_list = arch['ops_list']
skip_list = arch['skip_list'] 

args = obtain_search_args()
cfg = Config()

model = Deeplab_v3plus(cfg=cfg, args=args, alphas=ops_list, betas=net_arch, gammas=skip_list).cuda()
# model = Deeplab_v3plus(cfg).cuda()
model.train()
# model.eval()

x = torch.randn(2, 3, 128, 128).cuda()

from thop import profile

params, flops = profile(model=model, inputs=(x,))

print(params)
print(flops)