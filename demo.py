# from cityscapes import CityScapes, collate_fn2

# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from configs.configurations import Config


# if __name__ == '__main__':
#     cfg = Config()
#     cfg.datapth = r'D:\datasets\cityscapes'
#     ds = CityScapes(cfg, mode='train', num_copys=2)
#     # print(ds[0])

#     dl = DataLoader(ds,
#                     batch_size = 4,
#                     shuffle = True,
#                     num_workers = 4,
#                     collate_fn=collate_fn2,
#                     drop_last = True)
#     for im_lb in dl:
#         print(im_lb)
#         break

import torch
import torch.nn as nn


from cityscapes import CityScapes, collate_fn2
from loss import OhemCELoss, pgc_loss

from tqdm import tqdm
from torch.utils.data import DataLoader
from models.deeplabv3plus import Deeplab_v3plus
from configs.configurations import Config

if __name__ == "__main__":
    cfg = Config()
    cfg.datapth = r'D:\datasets\cityscapes'
    cfg.crop_size = (256, 256)
    ds = CityScapes(cfg, mode='train', num_copys=2)
    # print(ds[0])

    dl = DataLoader(ds,
                    batch_size = 1,
                    shuffle = True,
                    num_workers = 4,
                    collate_fn=collate_fn2,
                    drop_last = True)
    # for im_lb in dl:
    #     break

    net = Deeplab_v3plus(cfg)
    net.cuda()
    net.train()
    net = nn.DataParallel(net)
    diter = iter(dl)
    im, lb, overlap, flip = next(diter)
    lb = lb.cuda()
    im1, im2 = im[::2], im[1::2]
    logits1 = net(im1)
    logits2 = net(im2)
    outputs = []
    for f1, f2 in zip(logits1, logits2):
        outputs.append([f1, f2])
    n_min = cfg.ims_per_gpu*cfg.crop_size[0]*cfg.crop_size[1]//16
    criteria = OhemCELoss(thresh=cfg.ohem_thresh, n_min=n_min).cuda()
    Criterion = pgc_loss(use_pgc = [0,1,2], criteria=criteria)
    mse, sym_ce, mid_mse, mid_ce, mid_l1, ce = Criterion(outputs, overlap, flip, lb)
    loss = cfg.beta * sym_ce + ce
    gc_loss = sum(mid_mse)
    loss += cfg.alpha * gc_loss
    print(loss)

    # for i in range(100):
    #     #  with torch.no_grad():
    #     in_ten = torch.randn((1, 3, 768, 768)).cuda()
    #     logits = net(in_ten)
        
    #     for logit in logits:
    #         print(logit.size())
    #     break
