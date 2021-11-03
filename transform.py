#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import json
import random
import numpy as np


import torch
import torchvision.transforms as transforms


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h):
            return dict(im = im, lb = lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    )


    def test(self, im=None, ori_im=None, cur_cor=None, ori_cor=None, flip=1):
        crop_img = np.array(ori_im)
        print(crop_img.shape)
        crop_img = crop_img[ori_cor[0]:ori_cor[2], ori_cor[1]:ori_cor[3]]
        if flip == -1:
            crop_img = cv2.flip(crop_img, 1)

        print((crop_img == np.array(im)).all())
        im.show()
        # ori_im.show()
        ori_im = np.array(ori_im)
        print(ori_cor)
        ptLeftTop, ptRightBottom = (ori_cor[1], ori_cor[0]), (ori_cor[3], ori_cor[2])
        cv2.rectangle(ori_im, ptLeftTop, ptRightBottom, (0, 255, 0), 3, 4)
        cv2.imshow('img', ori_im)
        cv2.waitKey(0)


        exit()
        
        pass



    def mini_call(self, im, lb, flip, ori_im=None):
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h):
            return dict(im = im, lb = lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop_box = int(sw), int(sh), int(sw) + W, int(sh) + H
        im1 = im.crop(crop_box)
        lb1 = lb.crop(crop_box)

        # cur_cor = [0, 0, W, H]
        # ori_cor = [int(sh), int(sw), int(sw) + W, int(sh) + H]
        
        
        
        cur_cor = [0, 0, H, W]
        # print(flip)

        if flip == 1:
            ori_cor = [int(sh), int(sw), int(sh) + H, int(sw) + W]
        elif flip == -1:
            ori_cor = [int(sh), w - (int(sw) + W), int(sh) + H, w - (int(sw))]
        # self.test(im1, ori_im, cur_cor, ori_cor, flip)
        # exit()
        return im1, lb1, cur_cor, ori_cor



class Pair_RandomCrop(RandomCrop):
    def __init__(self, size, *args, **kwargs):
        super(Pair_RandomCrop, self).__init__(size, *args, **kwargs)
    


    def get_overlaps(self, cur_cors, ori_cors, flips):
        
        # print(cur_cors)
        # print(ori_cors)
        # print(flips)
        # exit()

        overlaps = []
        up = max(ori_cors[0][0], ori_cors[1][0])
        left = max(ori_cors[0][1], ori_cors[1][1])
        down = min(ori_cors[0][2], ori_cors[1][2])
        right = min(ori_cors[0][3], ori_cors[1][3])
        #overlap = [up, left, down, right]

        up_left = (up, left)
        down_right = (down, right)
        for i in range(len(cur_cors)):
            flip = flips[i]
            ori_cor = ori_cors[i]
            cur_cor = cur_cors[i]
            # print('cur_cor = {:}'.format(cur_cor))
            if flip == 1:
                size_y, size_x = cur_cor[2] - cur_cor[0], cur_cor[3] - cur_cor[1]
                _up_left = (round(cur_cor[0] + size_y * (up_left[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
                            round(cur_cor[1] + size_x * (up_left[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1])))
                _down_right = (round(cur_cor[0] + size_y * (down_right[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
                                round(cur_cor[1] + size_x * (down_right[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1])))
            elif flip == -1:

                size_y, size_x = cur_cor[2] - cur_cor[0], cur_cor[3] - cur_cor[1]
                _up_left = (round(cur_cor[0] + size_y * (up_left[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
                            round(cur_cor[1] + size_x * (1 - (down_right[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1]))))
                _down_right = (round(cur_cor[0] + size_y * (down_right[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
                                round(cur_cor[1] + size_x * (1 - (up_left[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1]))))
            overlaps.append([_up_left, _down_right])
        return overlaps#, overlap


        # overlaps = []
        # up = max(ori_cors[0][0], ori_cors[1][0])
        # left = max(ori_cors[0][1], ori_cors[1][1])
        # down = min(ori_cors[0][2], ori_cors[1][2])
        # right = min(ori_cors[0][3], ori_cors[1][3])
        # up_left = (up, left)
        # down_right = (down, right)

        # for i in range(len(cur_cors)):
        #     ori_cor = ori_cors[i]
        #     cur_cor = cur_cors[i]
        #     size_y, size_x = cur_cor[2] - cur_cor[0], cur_cor[3] - cur_cor[1]
        #     _up_left = (round(cur_cor[0] + size_y * (up_left[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
        #                 round(cur_cor[1] + size_x * (up_left[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1])))
        #     _down_right = (round(cur_cor[0] + size_y * (down_right[0] - ori_cor[0]) / (ori_cor[2] - ori_cor[0])),
        #                     round(cur_cor[1] + size_x * (down_right[1] - ori_cor[1]) / (ori_cor[3] - ori_cor[1])))
        #     overlaps.append([_up_left, _down_right])

        # return overlaps


    @staticmethod
    def good_sample(ori_cors):
        up = max(ori_cors[0][0], ori_cors[1][0])
        left = max(ori_cors[0][1], ori_cors[1][1])
        down = max(min(ori_cors[0][2], ori_cors[1][2]), up)
        right = max(min(ori_cors[0][3], ori_cors[1][3]), left)
        good_sample = True if (up < down and left < right) else False
        return good_sample


    @staticmethod
    def Rec(img, box, point_color = (0, 255, 0), thickness = 1, lineType = 4, show=False, crop=False):

        if crop:
            return img[box[0][0]:box[1][0], box[0][1]:box[1][1]]

        ptLeftTop = (box[1], box[0])
        ptRightBottom = (box[3], box[2])

        img = cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)


        if show:
            cv2.imshow('AlanWang', img)
            cv2.waitKey(0) # 显示 10000 ms 即 10s 后消失 
        return img



    def sample(self, ims, lbs, flips, ori_im=None):
        _ims = []
        _lbs = []
        cur_cors = []
        ori_cors = []

        for im, lb, flip in zip(ims, lbs, flips):
            im, lb, cur_cor, ori_cor = RandomCrop.mini_call(self, im, lb, flip, ori_im)
            _ims.append(im)
            _lbs.append(lb)
            cur_cors.append(cur_cor)
            ori_cors.append(ori_cor)

        overlaps = self.get_overlaps(cur_cors, ori_cors, flips)
        
        # ori_im = np.array(ori_im)
        # ori_im = self.Rec(ori_im, ori_cors[0], point_color=(0, 255, 0))
        # ori_im = self.Rec(ori_im, ori_cors[1], point_color=(255, 0, 0))
        # cv2.imwrite('ori_im.jpg', ori_im)



        # im1, im2 = np.array(_ims[0]), np.array(_ims[1])

        # # print(im1.shape)
        # # print(im2.shape)
        
        # # exit()



        # img = np.hstack([im1, im2])
        # cv2.imwrite('ori_stack.jpg', img)
        # im1 = self.Rec(im1, overlaps[0], crop=True)
        # im2 = self.Rec(im2, overlaps[1], crop=True)
        # if (im1 == im2).all():
        #     exit()
        # else:
        #     img = np.hstack([im1, im2])
        #     cv2.imwrite('stack.jpg', img)
        # exit()
        # # print(im1.shape)
        # # print(im2.shape)
        
        # # exit()



        # _img = np.array(ori_im)[overlap[0]:overlap[2], overlap[1]:overlap[3]]
        
        # cv2.imshow('img', _img)
        # cv2.waitKey(0)
        # exit()


        if self.good_sample(ori_cors):
            return dict(im=_ims, lb=_lbs, overlap=overlaps)
        else:
            return self.sample(ims, lbs, flips)

    def __call__(self, im_lbs):
        ims = im_lbs['im']
        lbs = im_lbs['lb']
        flips = im_lbs['flip']
        
        ori = im_lbs['ori']
        ori_im = ori['im']
        new_imlbs = self.sample(ims, lbs, flips, ori_im)

        # overlaps = new_imlbs['overlap']
        # _ims = new_imlbs['im']
        # im1, im2 = np.array(_ims[0]), np.array(_ims[1])
        # img = np.hstack([im1, im2])
        # cv2.imwrite('ori_stack.jpg', img)
        # im1 = self.Rec(im1, overlaps[0], crop=True)
        # im2 = self.Rec(im2, overlaps[1], crop=True)
        # if (im1 == im2).all():
        #     print('exit')
        #     exit()
        # else:
        #     img = np.hstack([im1, im2])
        #     cv2.imwrite('stack.jpg', img)


        im_lbs.update(new_imlbs)
        return im_lbs

class HorizontalFlip(object):
    def __init__(self, p = 0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )

    def call(self, im, lb):
        if random.random() > self.p:
            return im, lb, 1
        else:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            lb = lb.transpose(Image.FLIP_LEFT_RIGHT)
        return im, lb, -1


class Pair_HorizontalFlip(HorizontalFlip):
    def __init__(self, p=0.5, *args, **kwargs):
        super(Pair_HorizontalFlip, self).__init__(p, *args, **kwargs)

    def __call__(self, im_lb):
        ims, lbs = im_lb['im'], im_lb['lb']
        flips = []
        for i, (im,lb) in enumerate(zip(ims, lbs)):
            im, lb, is_flip = HorizontalFlip.call(self, im, lb)
            if is_flip:
                ims[i] = im
                lbs[i] = lb
            flips.append(is_flip)
        im_lb.update(flip=flips)
        return im_lb
                

class RandomScale(object):
    def __init__(self, scales = (1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                ), scale

class Pair_RandomScale(RandomScale):
    def __init__(self, scales = (1, ), img_size=None, *args, **kwargs):
        self.scales = scales
        self.H = img_size[0]
        self.W = img_size[1]


    def __call__(self, im_lb):
        ori = im_lb.copy()
        im_lb, scale = RandomScale.__call__(self, im_lb)
        im = im_lb['im']
        lb = im_lb['lb']
        ims = [im, im.copy()]
        lbs = [lb, lb.copy()]
        H, W = int(self.H * scale), int(self.W * scale)
        transform = [0, 0, self.H, self.W, 0, 0, H, W, scale, 0]
        im_lb.update(dict(im=ims, lb=lbs, transform = [transform, transform.copy()], ori=ori))
        return im_lb

class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im = im,
                    lb = lb,
                )

    def call(self, ims):
        new_ims = []
        for im in ims:
            r_brightness = random.uniform(self.brightness[0], self.brightness[1])
            r_contrast = random.uniform(self.contrast[0], self.contrast[1])
            r_saturation = random.uniform(self.saturation[0], self.saturation[1])
            im = ImageEnhance.Brightness(im).enhance(r_brightness)
            im = ImageEnhance.Contrast(im).enhance(r_contrast)
            im = ImageEnhance.Color(im).enhance(r_saturation)
            new_ims.append(im)
        return new_ims



class Pair_ColorJitter(ColorJitter):
    def __init__(self, brightness=None, contrast=None, saturation=None,  *args, **kwags):
        super(Pair_ColorJitter, self).__init__(brightness, contrast, saturation, *args, **kwags)


    def __call__(self, im_lb):
        ims = im_lb['im']
        ims = ColorJitter.call(self, ims)
        im_lb.update({'im': ims})
        return im_lb
        

class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb




class Pair_ToTensor(object):
    def __init__(self, to_tensor=None):
        self.to_tensor = to_tensor
        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



    def __call__(self, im_lb):
        ims = im_lb['im']
        lbs = im_lb['lb']
        for i, (im, lb) in enumerate(zip(ims, lbs)):
            im = self.to_tensor(im)
            lb = np.array(lb).astype(np.int64)[np.newaxis, :]
            lb = self.convert_labels(lb)
            lb = torch.from_numpy(lb)
            ims[i] = im
            lbs[i] = lb
        return im_lb



if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
