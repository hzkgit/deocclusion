import copy

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class L2LossWithIgnore(nn.Module):

    def __init__(self, ignore_value=None):
        super(L2LossWithIgnore, self).__init__()
        self.ignore_value = ignore_value

    def forward(self, input, target): # N1HW, N1HW
        if self.ignore_value is not None:
            target_area = target != self.ignore_value
            target = target.float()
            return (input[target_area] - target[target_area]).pow(2).mean()
        else:
            return (input - target.float()).pow(2).mean()


class MaskWeightedCrossEntropyLoss(nn.Module):

    def __init__(self, inmask_weight=5, outmask_weight=1):
        super(MaskWeightedCrossEntropyLoss, self).__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight

    def forward(self, predict, target, mask):
        '''
        predict: NCHW,(4,2,256,256)
        target: NHW,(4,256,256)
        mask: NHW,(4,256,256)
        '''
        n, c, h, w = predict.size()

        mask = mask.byte()  # .byte()方法被用来将数组中的元素转换为8位无符号整型，即字节型(uint8)数据类型
        target_inmask = target[mask]  # 若mask对应位置元素>0则target对应位置元素保留，否则target对应位置元素去掉（提取mask对应位置像素）(4,256,256)=>(26972,)
        target_outmask = target[~mask]  # ~用作取反操作符,保留target所有位置元素 (4,256,256)=>(262144,)
        # temp = copy.deepcopy(predict.cpu().detach())
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()  # (4,2,256,256)=>(4,256,256,2) (n,c,h,w)=>(n,h,w,c)  2个 256*256矩阵a和b 变成256个 256*2的矩阵 第一个矩阵为(a1:b1,a2:b2...,a256,b256)
        # 每一组predict、target、erase都用俩种覆盖方式（第一种：erase盖在modal上；第二种：predict盖在erase上）
        # 然后L1和L2做交叉熵
        predict_inmask = predict[mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)  # (4,256,256)=>(4,256,256,1)=>(4,256,256,2)(提取mask对应位置像素)=>(53944,)=>(26972,2)
        predict_outmask = predict[(~mask).view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)  # (4,256,256)=>(4,256,256,1)=>(4,256,256,2)=>(524288,)=>(262144,2)
        loss_inmask = nn.functional.cross_entropy(  # L1
            predict_inmask, target_inmask.long(), size_average=False)  # size_average是否计算损失值的平均
        loss_outmask = nn.functional.cross_entropy(  # L2
            predict_outmask, target_outmask.long(), size_average=False)
        loss = (self.inmask_weight * loss_inmask + self.outmask_weight * loss_outmask) / (n * h * w)  # （5*L1损失值+1*L2损失值）/4*256*256
        return loss


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict
