#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import floor




class new_ssp_loss(nn.Module):
    def __init__(self, exclusive=True, ignore_index=255, criteria=None):
        super(new_ssp_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)# if criteria is None else criteria
        self.mse_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.ignore_index = ignore_index

    def forward(self, outputs, overlap, flips, labels):
        output1, output2 = outputs
        N = output1.shape[0]
        mse = 0
        ce_1_2 = 0
        ce_2_1 = 0
        ex_labels = labels.detach().clone()

        # for i in range(N):
        for i in range(N):
            shape_1 = (overlap[i][0][1][0] - overlap[i][0][0][0], overlap[i][0][1][1] - overlap[i][0][0][1])
            shape_2 = (overlap[i][1][1][0] - overlap[i][1][0][0], overlap[i][1][1][1] - overlap[i][1][0][1])
            img_1 = output1[i, :, overlap[i][0][0][0]:overlap[i][0][1][0], overlap[i][0][0][1]:overlap[i][0][1][1]]
            img_2 = output2[i, :, overlap[i][1][0][0]:overlap[i][1][1][0], overlap[i][1][0][1]:overlap[i][1][1][1]]



            ex_labels[2 * i, overlap[i][0][0][0]:overlap[i][0][1][0], overlap[i][0][0][1]:overlap[i][0][1][1]] = self.ignore_index
            ex_labels[2 * i + 1, overlap[i][1][0][0]:overlap[i][1][1][0], overlap[i][1][0][1]:overlap[i][1][1][1]] = self.ignore_index

            if flips[i] == -1:
                img_2 = torch.flip(img_2, [2])

            if ((shape_1[0] < 1) or (shape_1[1] < 1) or (shape_2[0] < 1) or (shape_2[1] < 1)):
                mse_loss = 0
                ce_loss_1_2 = 0
                ce_loss_2_1 = 0
            else:
                img_1_label = img_1.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                img_2_label = img_2.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                mse_loss = self.mse_loss(img_1, img_2)

                ce_loss_1_2 = self.ce_loss(img_1.unsqueeze(0), img_2_label)
                ce_loss_2_1 = self.ce_loss(img_2.unsqueeze(0), img_1_label)
                sym_ce_loss = 0.5 * ce_loss_1_2 + 0.5 * ce_loss_2_1
            mse += mse_loss
            ce_1_2 += ce_loss_1_2
            ce_2_1 += ce_loss_2_1

        mse /= N
        ce_1_2 /= N
        ce_2_1 /= N
        sym_ce = 0.5 * (ce_1_2 + ce_2_1)
        label1 = labels[::2]
        label2 = labels[1::2]
        exlabel1 = ex_labels[::2]
        exlabel2 = ex_labels[1::2]
        Labels = torch.cat([label1, label2], dim=0).detach()
        ex_labels = torch.cat([exlabel1, exlabel2], dim=0).detach()
        Output = torch.cat([outputs[0], outputs[1]], dim=0)

        ce = self.ce_loss(Output, Labels)
        #
        ex_ce = self.ce_loss(Output, ex_labels)
        return mse, ce_1_2, ce_2_1, sym_ce, ce


class ssp_loss_inner(new_ssp_loss):
    def __init__(self, criteria=None) -> None:
        super(ssp_loss_inner, self).__init__(exclusive=True, ignore_index=255, criteria=criteria)
        self.criteria = criteria

    def forward(self, outputs, overlap, flips, downsamples=1):
        len_img = outputs[0].shape[0]
        mse = 0
        l1 = 0
        ce_1_2 = 0
        ce_2_1 = 0

        # overlap_new = overlap.copy()
        overlap_new = np.zeros((len_img, 2, 2, 2), dtype=np.int)


        if downsamples != 1:
            for i in range(len_img):
                for j in range(2):
                    if (j == 0):
                        for k in range(2):
                            for l in range(2):
                                overlap_new[i][j][k][l] = floor(overlap[i][j][k][l] / downsamples)
                        h = overlap_new[i][j][1][0] - overlap_new[i][j][0][0]
                        w = overlap_new[i][j][1][1] - overlap_new[i][j][0][1]
                        size = (h, w)
                #       print('h:', h, 'w:', w)

                    elif (j == 1):
                        for k in range(2):
                            for l in range(2):
                                if k == 0:
                                    overlap_new[i][j][k][l] = (overlap[i][j][k][l] // downsamples)
                                else:
                                    overlap_new[i][j][k][l] = overlap_new[i][j][0][l] + size[l]

        for i in range(len_img):
            shape_1 = (overlap_new[i][0][1][0] - overlap_new[i][0][0][0], overlap_new[i][0][1][1] - overlap_new[i][0][0][1])
            shape_2 = (overlap_new[i][1][1][0] - overlap_new[i][1][0][0], overlap_new[i][1][1][1] - overlap_new[i][1][0][1])
            img_1 = outputs[0][:, overlap_new[i][0][0][0]:overlap_new[i][0][1][0], overlap_new[i][0][0][1]:overlap_new[i][0][1][1]]
            img_2 = outputs[1][:, overlap_new[i][1][0][0]:overlap_new[i][1][1][0], overlap_new[i][1][0][1]:overlap_new[i][1][1][1]]

            if flips[i] == -1:
                img_2 = torch.flip(img_2, [2])
    
            if ((shape_1[0] < 1) or (shape_1[1] < 1) or (shape_2[0] < 1) or (shape_2[1] < 1)):
                mse_loss = 0
                ce_loss_1_2 = 0
                ce_loss_2_1 = 0
                l1_loss = 0
            else:
                img_1_label = img_1.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                img_2_label = img_2.argmax(dim=0).unsqueeze(0).requires_grad_(False)
                mse_loss = self.mse_loss(img_1, img_2)
                l1_loss = self.L1_loss(img_1, img_2)

                ce_loss_1_2 = self.ce_loss(img_1.unsqueeze(0), img_2_label)
                ce_loss_2_1 = self.ce_loss(img_2.unsqueeze(0), img_1_label)
                mse += mse_loss

            mse += mse_loss
            l1 += l1_loss
            ce_1_2 += ce_loss_1_2
            ce_2_1 += ce_loss_2_1

        mse /= len_img
        ce_1_2 /= len_img
        ce_2_1 /= len_img
        l1 /= len_img
        sym_ce = 0.5 * (ce_1_2 + ce_2_1)
        return mse, sym_ce, l1

class pgc_loss(ssp_loss_inner):
    def __init__(self, use_pgc=[0, 1, 2], down_rate=[16, 16, 4], criteria=None):
        super(pgc_loss, self).__init__(criteria=criteria)
        self.use_pgc = use_pgc
        self.down_rate = dict(zip(use_pgc, down_rate))

    def forward(self, outputs, overlap, flips, labels):
        mse, _, _, sym_ce, ce = new_ssp_loss.forward(self, outputs[-1], overlap, flips, labels)
        mid_mse = []
        mid_ce = []
        mid_l1 = []
        for i in self.use_pgc:
            down_rate = self.down_rate[i]
            mse1, sym_ce1, l11 = ssp_loss_inner.forward(self, outputs[i], overlap, flips, down_rate)
            mid_mse.append(mse1)
            mid_ce.append(sym_ce1)
            mid_l1.append(l11)

        return mse, sym_ce, mid_mse, mid_ce, mid_l1, ce


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)

        labels = labels.contiguous().view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu==self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min]<self.thresh else sorteds[self.n_min]
            labels[picks>thresh] = self.ignore_lb
        ## TODO: here see if torch or numpy is faster
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss


if __name__ == '__main__':
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, 10, 10] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    loss.backward()
