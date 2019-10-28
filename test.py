
import torch
from configs.configurations import Config


from models.deeplabv3plus_naive_connection import Deeplab_v3plus

cfg = Config()
model = Deeplab_v3plus(cfg)

x = torch.randn(2, 3, 129, 129)

print(model(x).shape)
