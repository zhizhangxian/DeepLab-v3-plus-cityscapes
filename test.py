
import torch
from configs.configurations import Config


from models.panoptic_deeplab import Deeplab_v3plus

cfg = Config()
model = Deeplab_v3plus(cfg)

x = torch.randn(2, 3, 128, 128)

print(model(x).shape)
