import numpy as np

import torch
import torch.nn as nn

import utils
import inference as infer
from . import SingleStageModel
from . import MaskWeightedCrossEntropyLoss
import torchvision.utils as vutils

from PIL import Image
import matplotlib.pyplot as plt

import pdb


class PartialCompletionMask(SingleStageModel):

    def __init__(self, params, load_pretrain=None, dist_model=False):
        super(PartialCompletionMask, self).__init__(params, dist_model)
        self.params = params
        self.use_rgb = params['use_rgb']

        # loss
        self.criterion = MaskWeightedCrossEntropyLoss(
            inmask_weight=params['inmask_weight'],
            outmask_weight=1.)

    def set_input(self, rgb=None, mask=None, eraser=None, target=None):
        self.rgb = rgb.cuda()
        self.mask = mask.cuda()
        self.eraser = eraser.cuda()
        self.target = target.cuda()

    def evaluate(self, image, inmodal, category, bboxes, amodal, gt_order_matrix, input_size):
        order_method = self.params.get('order_method', 'ours')
        # order
        if order_method == 'ours':
            order_matrix = infer.infer_order2(
                self, image, inmodal, category, bboxes,
                use_rgb=self.use_rgb,
                th=self.params['inference']['positive_th_order'],
                dilate_kernel=self.params['inference'].get('dilate_kernel_order', 0),
                input_size=input_size,
                min_input_size=16,
                interp=self.params['inference']['order_interp'])
        elif order_method == 'hull':
            order_matrix = infer.infer_order_hull(inmodal)
        elif order_method == 'area':
            order_matrix = infer.infer_order_area(inmodal, above=self.params['above'])
        elif order_method == 'yaxis':
            order_matrix = infer.infer_order_yaxis(inmodal)
        else:
            raise Exception("No such method: {}".format(order_method))

        gt_order_matrix = infer.infer_gt_order(inmodal, amodal)
        allpair_true, allpair, occpair_true, occpair, show_err = infer.eval_order(
            order_matrix, gt_order_matrix)

        # amodal
        amodal_method = self.params.get('amodal_method', 'ours')
        if amodal_method == 'ours':
            amodal_patches_pred = infer.infer_amodal(
                self, image, inmodal, category, bboxes,
                order_matrix, use_rgb=self.use_rgb,
                th=self.params['inference']['positive_th_amodal'],
                dilate_kernel=self.params['inference'].get('dilate_kernel_amodal', 0),
                input_size=input_size,
                min_input_size=16, interp=self.params['inference']['amodal_interp'],
                order_grounded=self.params['inference']['order_grounded'])
            amodal_pred = infer.patch_to_fullimage(
                amodal_patches_pred, bboxes,
                image.shape[0], image.shape[1],
                interp=self.params['inference']['amodal_interp'])
        elif amodal_method == 'hull':
            amodal_pred = np.array(infer.infer_amodal_hull(
                inmodal, bboxes, order_matrix,
                order_grounded=self.params['inference']['order_grounded']))
        elif amodal_method == 'raw':
            amodal_pred = inmodal  # evaluate raw
        else:
            raise Exception("No such method: {}".format(amodal_method))

        intersection = ((amodal_pred == 1) & (amodal == 1)).sum()
        union = ((amodal_pred == 1) | (amodal == 1)).sum()
        target = (amodal == 1).sum()

        return allpair_true, allpair, occpair_true, occpair, intersection, union, target

    def forward_only(self, ret_loss=True):
        with torch.no_grad():
            if self.use_rgb:
                output = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
            else:
                output = self.model(torch.cat([self.mask, self.eraser], dim=1))
            if output.shape[2] != self.mask.shape[2]:
                output = nn.functional.interpolate(
                    output, size=self.mask.shape[2:4],
                    mode="bilinear", align_corners=True)
        comp = output.argmax(dim=1, keepdim=True).float()  # # output:(4,2,256,256)=>(4,1,256,256)
        comp[self.eraser == 0] = (self.mask > 0).float()[
            self.eraser == 0]  # comp[self.eraser == 0]获取comp中eraser中所有为True的像素位置

        vis_combo = (self.mask > 0).float()
        vis_combo[self.eraser == 1] = 0.5  # 将模态掩码图中与遮挡物像素为1的点的相同位置的像素设为0.5（灰度化）便于查看
        vis_target = self.target.cpu().clone().float()
        if vis_target.max().item() == 255:  # 这里没过
            vis_target[vis_target == 255] = 0.5
        vis_target = vis_target.unsqueeze(1)
        if self.use_rgb:
            cm_tensors = [self.rgb]
        else:
            cm_tensors = []
        ret_tensors = {'common_tensors': cm_tensors,
                       'mask_tensors': [self.mask, vis_combo, comp,
                                        vis_target]}  # mask:模态掩码 vis_combo:遮挡对象置灰（便于观察） comp:预测结果 vis_target：目标
        if ret_loss:
            loss = self.criterion(output, self.target, self.eraser.squeeze(1)) / self.world_size  # 求损失值
            return ret_tensors, {'loss': loss}
        else:
            return ret_tensors

    def step(self):

        test = torch.cat([self.mask, self.eraser], dim=1)
        tt1 = test[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
        tt2 = test[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
        test2 = torch.cat((tt1, tt2), dim=0)
        grid2 = vutils.make_grid(test2,
                                nrow=4,
                                normalize=True,
                                range=(0, 255),
                                scale_each=False)

        img_pil2 = Image.fromarray(np.uint8(grid2.cpu().permute(1, 2, 0) * 255))  # (3, 518, 518)
        plt.imshow(img_pil2)
        plt.show()

        if self.use_rgb:
            output = self.model(torch.cat([self.mask, self.eraser], dim=1), self.rgb)
        else:
            output = self.model(torch.cat([self.mask, self.eraser], dim=1))

        # # 输出结果可视化
        # for i in range(4):
        #     outputTemp = output.cpu().detach().numpy()
        #     temp = outputTemp[i][0]
        #     img_pil = Image.fromarray(np.uint8(temp))
        #     plt.imshow(img_pil)
        #     plt.show()
        #
        #     temp = outputTemp[i][1]
        #     img_pil = Image.fromarray(np.uint8(temp))
        #     plt.imshow(img_pil)
        #     plt.show()

        # temp = nn.functional.normalize(output, p=2, dim=None)  # (4,2,256,256)
        # temp = torch.clamp(temp, min=0, max=1)

        temp_1 = output.argmax(dim=1, keepdim=True).cpu().detach().numpy()
        temp_2 = output.cpu().detach().numpy()

        comp = output.argmax(dim=1, keepdim=True).float()  # (4,2,256,256)=>(4,1,256,256)
        comp[self.eraser == 0] = (self.mask > 0).float()[self.eraser == 0]

        # output:(4,2,256,256)
        t1 = output[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
        t2 = output[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
        t3 = self.target.unsqueeze(1).repeat(1, 3, 1, 1)
        t4 = self.eraser.repeat(1, 3, 1, 1)
        t5 = comp.repeat(1, 3, 1, 1)

        images_tensor = torch.cat((t1, t2, t3, t4, t5), dim=0)

        grid = vutils.make_grid(images_tensor,
                                nrow=4,
                                normalize=True,
                                range=(0, 255),
                                scale_each=True)

        # vutils.save_image(grid, 'grid_image.png')
        img_pil = Image.fromarray(np.uint8(grid.cpu().permute(1, 2, 0) * 255))  # (3, 518, 518)
        plt.imshow(img_pil)
        plt.show()

        loss = self.criterion(output, self.target, self.eraser.squeeze(1)) / self.world_size  # 这一步出现了问题
        self.optim.zero_grad()
        loss.backward()
        utils.average_gradients(self.model)
        self.optim.step()
        return {'loss': loss}
