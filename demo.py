# from cityscapes import CityScapes, collate_fn2

# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from configs.configurations import Config



# # def get_overlaps(cur_cors, ori_cors):
# #     overlaps = []
# #     up = max(ori_cors[0][0], ori_cors[1][0])
# #     left = max(ori_cors[0][1], ori_cors[1][1])
# #     down = min(ori_cors[0][2], ori_cors[1][2])
# #     right = min(ori_cors[0][3], ori_cors[1][3])
# #     up_left = (up, left)
# #     down_right = (down, right)

# #     for i in range(len(cur_cors)):
# #         ori_cor = ori_cors[i]
# #         cur_cor = cur_cors[i]
# #         size_y, size_x = cur_cor[2] - cur_cor[0], cur_cor[3] - cur_cor[1]
# #         _up_left = (round(cur_cor[0] + size_y * (up_left[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
# #                     round(cur_cor[1] + size_x * (up_left[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1])))
# #         _down_right = (round(cur_cor[0] + size_y * (down_right[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
# #                         round(cur_cor[1] + size_x * (down_right[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1])))
# #         overlaps.append([_up_left, _down_right])

# #     return overlaps



# def get_hw(overlap):
#     up, left = overlap[0]
#     down, right = overlap[1]
#     h = down - up
#     w = right - left

#     return h, w

# if __name__ == '__main__':
#     cfg = Config()
#     # cfg.datapth = r'D:\datasets\cityscapes'
#     ds = CityScapes(cfg, mode='train', num_copys=2)
#     # for i in range(100):
#     #     sample = ds[0]
#     #     overlap1, overlap2 = sample['overlap']


#     #     h1, w1 = get_hw(overlap1)
#     #     h2, w2 = get_hw(overlap2)
#     #     print((h1 == h2) and (w1 == w2))
#     #     print(h1, w1)

#     dl = DataLoader(ds,
#                     batch_size = 4,
#                     shuffle = True,
#                     num_workers = 4,
#                     collate_fn=collate_fn2,
#                     drop_last = True)
#     for im_lb in dl:
#         im = im_lb[0]
#         lb = im_lb[1]
#         print(im.shape)
#         print(lb.shape)
#         break

import torch
import torch.nn as nn

import numpy as np

from cityscapes import CityScapes, collate_fn2
from loss import OhemCELoss, pgc_loss

from tqdm import tqdm
from torch.utils.data import DataLoader
# from models.deeplabv3plus import Deeplab_v3plus
from configs.configurations import Config

import cv2


def Rec(img, box, point_color = (0, 255, 0), thickness = 1, lineType = 4, show=False, crop=False):

    if show:
        ptLeftTop = (box[0][1], box[0][0])
        ptRightBottom = (box[1][1], box[1][0])

        img = cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        cv2.imshow('AlanWang', img)
        cv2.waitKey(0) # 显示 10000 ms 即 10s 后消失 


    if crop:
        return img[box[0][0]:box[1][0], box[0][1]:box[1][1]]
    return img




# def Rec(img, box, point_color = (0, 255, 0), thickness = 1, lineType = 4):

#     ptLeftTop, ptRightBottom = box[0], box[1]
    
#     img = cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
    
#     cv2.imshow('AlanWang', img)
#     cv2.waitKey(0) # 显示 10000 ms 即 10s 后消失 
#     return img


if __name__ == "__main__":
    cfg = Config()
    cfg.datapth = r'D:\datasets\cityscapes'
    cfg.crop_size = (384, 384)
    ds = CityScapes(cfg, mode='train', num_copys=2)


    # new_imlbs = ds[0]
    # overlaps = new_imlbs['overlap']
    # flips = new_imlbs['flip']
    # flip = flips[0] * flips[1]
    # _ims = new_imlbs['im']
    # im1, im2 = np.array(_ims[0]), np.array(_ims[1])
    # img = np.hstack([im1, im2])
    # cv2.imwrite('ori_stack.jpg', img)
    # im1 = Rec(im1, overlaps[0], crop=True)
    # im2 = Rec(im2, overlaps[1], crop=True)
    # if flip == -1:
    #     im2 = cv2.flip(im2, 1)
    # if (im1 == im2).all():
    #     print('exit')
    #     exit()
    # else:
    #     print('noop')
    #     img = np.hstack([im1, im2])
    #     cv2.imwrite('stack.jpg', img)

    # exit()







    # sample = ds[0]
    # ims, lbs, overlap = sample['im'], sample['lb'], sample['overlap']#, sample['flip'],
    # try:
    #     flip = sample['flip']
    # except:
    #     flip = 1
    # # print(overlap)
    # im1, im2 = ims

    # im1 = np.array(im1)
    # im2 = np.array(im2)

    # print(flip)



    # # box1 = [(overlap[0][0][0], overlap[0][0][1]), (overlap[0][1][0], overlap[0][1][1])]
    # box1 = overlap[0]
    # print(box1)
    # # box2 = [(overlap[1][0][0], overlap[1][0][1]), (overlap[1][1][0], overlap[1][1][1])]
    # box2 = overlap[1]
    # print(box2)


    # im1 = Rec(im1, box1, show=False, crop=True)
    # im2 = Rec(im2, box2, show=False, crop=True)
    # if flip[0] * flip[1] == -1:
    #     im2 = cv2.flip(im2, 1)
    # if (im1 == im2).all():
    #     exit()
    # img = np.hstack([im1, im2])
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # exit()

    



    dl = DataLoader(ds,
                    batch_size = 2,
                    shuffle = False,
                    num_workers = 2,
                    collate_fn=collate_fn2,
                    drop_last = True)
    for im_lb in dl:
        break


    diter = iter(dl)
    im, lb, overlap, flip = next(diter)
    print(flip[0])

    def crop(lb, overlap):
        return lb[overlap[0][0]:overlap[1][0], overlap[0][1]:overlap[1][1]]

    lb = lb.squeeze(1)
    lb1 = crop(lb[0], overlap[0][0])
    lb2 = crop(lb[1], overlap[0][1])
    if flip[0] == -1:
        lb2 = torch.flip(lb2, [1])

    print(lb1.shape)
    print(lb2.shape)
    
    print(lb.shape)
    print(overlap)

    print((lb1 == lb2).all())


    # # box1 = [(overlap[0][0][0], overlap[0][0][1]), (overlap[0][1][0], overlap[0][1][1])]
    # box1 = overlap[0]
    # print(box1)
    # # box2 = [(overlap[1][0][0], overlap[1][0][1]), (overlap[1][1][0], overlap[1][1][1])]
    # box2 = overlap[1]
    # print(box2)

    # net = Deeplab_v3plus(cfg)
    # net.cuda()
    # net.train()
    # net = nn.DataParallel(net)
    # diter = iter(dl)
    # im, lb, overlap, flip = next(diter)


    



    # lb = lb.cuda()
    # lb = lb.squeeze(1)
    # im1, im2 = im[::2], im[1::2]
    # logits1 = net(im1)
    # logits2 = net(im2)
    # outputs = []
    # for f1, f2 in zip(logits1, logits2):
    #     outputs.append([f1, f2])
    # n_min = cfg.ims_per_gpu*cfg.crop_size[0]*cfg.crop_size[1]//16
    # criteria = OhemCELoss(thresh=cfg.ohem_thresh, n_min=n_min).cuda()
    # Criterion = pgc_loss(use_pgc = [0,1,2], criteria=criteria)
    # mse, sym_ce, mid_mse, mid_ce, mid_l1, ce = Criterion(outputs, overlap, flip, lb)
    # loss = cfg.beta * sym_ce + ce
    # gc_loss = sum(mid_mse)
    # loss += cfg.alpha * gc_loss
    # loss.backward()
    # print(loss)

    # for i in range(100):
    #     #  with torch.no_grad():
    #     in_ten = torch.randn((1, 3, 768, 768)).cuda()
    #     logits = net(in_ten)
        
    #     for logit in logits:
    #         print(logit.size())
    #     break
