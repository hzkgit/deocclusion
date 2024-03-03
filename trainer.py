import os
import cv2
import time
import numpy as np

import torch
import torch.optim
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import models
import utils
import datasets
#from dataset import ImageRawDataset, PartialCompEvalDataset, PartialCompDataset
import inference as infer
import pdb

from PIL import Image
import matplotlib.pyplot as plt


class Trainer(object):

    def __init__(self, args):

        # get rank
        self.world_size = dist.get_world_size()  # 1
        self.rank = dist.get_rank()  # 0

        if self.rank == 0:
            # mkdir path 创建相关文件夹
            if not os.path.exists('{}/events'.format(args.exp_path)):
                os.makedirs('{}/events'.format(args.exp_path))
            if not os.path.exists('{}/images'.format(args.exp_path)):
                os.makedirs('{}/images'.format(args.exp_path))
            if not os.path.exists('{}/logs'.format(args.exp_path)):
                os.makedirs('{}/logs'.format(args.exp_path))
            if not os.path.exists('{}/checkpoints'.format(args.exp_path)):
                os.makedirs('{}/checkpoints'.format(args.exp_path))

            # logger 训练日志相关
            if args.trainer['tensorboard']:
                try:
                    from tensorboardX import SummaryWriter
                except:
                    raise Exception("Please switch off \"tensorboard\" "
                                    "in your config file if you do not "
                                    "want to use it, otherwise install it.")
                self.tb_logger = SummaryWriter('{}/events'.format(
                    args.exp_path))
            else:
                self.tb_logger = None
            if args.validate:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_offline_val.txt'.format(args.exp_path))
            else:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_train.txt'.format(args.exp_path))  # 定义训练日志保存位置

        # create model 创建模型（algo就是PartialCompletionMask模型）、传入config中配置的模型超参数
        self.model = models.__dict__[args.model['algo']](
            args.model, load_pretrain=args.load_pretrain, dist_model=True)

        # optionally resume from a checkpoint 是否从检查点恢复上次训练
        assert not (args.load_iter is not None and args.load_pretrain is not None), \
            "load_iter and load_pretrain are exclusive."
        if args.load_iter is not None:
            self.model.load_state("{}/checkpoints".format(args.exp_path),
                                  args.load_iter, args.resume)
            self.start_iter = args.load_iter
        else:
            self.start_iter = 0
        self.curr_step = self.start_iter

        # lr scheduler & datasets，trainval_dataset对应的是datasets/PartialCompDataset，在这一步获取rgb（原图）、mask（非模态掩码图）、reaser（遮挡对象图）、traget（目标还原图）
        trainval_class = datasets.__dict__[args.data['trainval_dataset']]

        if not args.validate:  # train
            # utils.StepLRScheduler是torch.optim模块下的一个学习率调度器，学习率调度器的主要作用是调整模型的学习率，使其在训练过程中能够有特定的变化，而不是一成不变。这样可以更好地优化模型的训练效果6。
            self.lr_scheduler = utils.StepLRScheduler(
                self.model.optim,
                args.model['lr_steps'],  # [32000, 48000]，学习率会在第32000个和第48000个迭代时分别调整一次
                args.model['lr_mults'],  # 每次调整学习率后的新学习率是当前学习率的多少倍
                args.model['lr'],  # 初始学习率
                args.model['warmup_lr'],
                args.model['warmup_steps'],
                last_iter=self.start_iter - 1)  # 表示上一次迭代的次数，用于判断当前迭代是否已经超过了学习率调整的迭代次数

            train_dataset = trainval_class(args.data, 'train')  # 初始化数据集
            train_sampler = utils.DistributedGivenIterationSampler(  # DistributedGivenIterationSampler：用于在分布式环境中进行数据采样 采样器中含有打乱顺序的240000（轮次*batch_size）个序号
                train_dataset,
                args.model['total_iter'],  # 总迭代次数56000
                args.data['batch_size'],   # 32
                last_iter=self.start_iter - 1)  # 表示上一次迭代的次数
            self.train_loader = DataLoader(train_dataset,  # 创建数据加载器
                                           batch_size=args.data['batch_size'],
                                           shuffle=False,
                                           num_workers=args.data['workers'],
                                           pin_memory=False,
                                           sampler=train_sampler)

        val_dataset = trainval_class(args.data, 'val')
        val_sampler = utils.DistributedSequentialSampler(val_dataset)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.data['batch_size_val'],
            shuffle=False,
            num_workers=args.data['workers'],
            pin_memory=False,
            sampler=val_sampler)

        self.args = args

    def run(self):

        # offline validate
        if self.args.validate:
            self.validate('off_val')  # 不走这里
            return

        if self.args.trainer['initial_val']:
            self.validate('on_val')  # 走这里

        # train
        self.train()

    def train(self):

        btime_rec = utils.AverageMeter(10)
        dtime_rec = utils.AverageMeter(10)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)  # 这段代码的意思就是设置recorder['loss']=utils.AverageMeter(10)

        self.model.switch_to('train')

        end = time.time()
        for i, inputs in enumerate(self.train_loader):
            self.curr_step = self.start_iter + i
            self.lr_scheduler.step(self.curr_step)
            curr_lr = self.lr_scheduler.get_lr()[0]

            # measure data loading time
            dtime_rec.update(time.time() - end)

            self.model.set_input(*inputs)  # 这一步将rgb（原图）、mask（模态掩码图）、reaser（遮挡对象图）、traget（目标还原图）输入模型
            loss_dict = self.model.step()  # 这一步出问题了  用于调用PartialCompletionMask模型的step()方法,获取损失值
            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item())

            btime_rec.update(time.time() - end)
            end = time.time()

            self.curr_step += 1

            # logging
            if self.rank == 0 and self.curr_step % self.args.trainer[
                    'print_freq'] == 0:  # 每100次输出一次
                loss_str = ""
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar('lr', curr_lr, self.curr_step)
                for k in recorder.keys():
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar('train_{}'.format(k),
                                                  recorder[k].avg,
                                                  self.curr_step)
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                        k, loss=recorder[k])  # k='loss' recoder['loss']获取损失信息，val总损失值 avg平均损失值（10个值的平均），.4g表示保留4位小数

                self.logger.info(
                    'Iter: [{0}/{1}]\t'.format(self.curr_step,
                                               len(self.train_loader)) +
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        batch_time=btime_rec) +
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        data_time=dtime_rec) + loss_str +
                    'lr {lr:.2g}'.format(lr=curr_lr))

            # save
            if (self.rank == 0 and
                (self.curr_step % self.args.trainer['save_freq'] == 0 or
                 self.curr_step == self.args.model['total_iter'])):
                self.model.save_state(
                    "{}/checkpoints".format(self.args.exp_path),
                    self.curr_step)

            # validate
            if (self.curr_step % self.args.trainer['val_freq'] == 0 or
                self.curr_step == self.args.model['total_iter']):
                self.validate('on_val')
            # self.validate('on_val')

    def validate(self, phase):
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('eval')

        end = time.time()
        all_together = []
        for i, inputs in enumerate(self.val_loader):  # inputs,原图(4,3,256,256)、模态掩码图、遮挡物掩码图、目标还原图
            if ('val_iter' in self.args.trainer and
                    self.args.trainer['val_iter'] != -1 and
                    i == self.args.trainer['val_iter']):
                break

            dtime_rec.update(time.time() - end)

            self.model.set_input(*inputs)
            tensor_dict, loss_dict = self.model.forward_only()
            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item())
            btime_rec.update(time.time() - end)
            end = time.time()

            # tb visualize
            if self.rank == 0:
                disp_start = max(self.args.trainer['val_disp_start_iter'], 0)
                disp_end = min(self.args.trainer['val_disp_end_iter'], len(self.val_loader))
                if (i >= disp_start and i < disp_end):  # 当i=0时进入
                    all_together.append(
                        utils.visualize_tensor(tensor_dict,
                        self.args.data.get('data_mean', [0,0,0]),
                        self.args.data.get('data_std', [1,1,1])))
                if (i == disp_end - 1 and disp_end > disp_start):  # 当i=0时进入
                    all_together = torch.cat(all_together, dim=2)
                    grid = vutils.make_grid(all_together,  # make_grid函数的作用是将多个图像合并成一个数组，并在一个图像中显示出来
                                            nrow=1,
                                            normalize=True,
                                            range=(0, 255),
                                            scale_each=False)
                    if self.tb_logger is not None:
                        self.tb_logger.add_image('Image_' + phase, grid,
                                                 self.curr_step)
                    cv2.imwrite("{}/images/{}_{}.png".format(
                        self.args.exp_path, phase, self.curr_step),
                        grid.permute(1, 2, 0).numpy()*255)  # 自己添加，不乘以255相识为全黑

        # logging
        if self.rank == 0:
            loss_str = ""
            for k in recorder.keys():
                if self.tb_logger is not None and phase == 'on_val':
                    self.tb_logger.add_scalar('val_{}'.format(k),
                                              recorder[k].avg,
                                              self.curr_step)
                loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                    k, loss=recorder[k])

            self.logger.info(
                'Validation Iter: [{0}]\t'.format(self.curr_step) +
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_time=btime_rec) +
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    data_time=dtime_rec) + loss_str)

        self.model.switch_to('train')
