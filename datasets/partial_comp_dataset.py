import numpy as np
try:
    import mc
except Exception:
    pass
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
from . import reader

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class PartialCompDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        if self.dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(config['{}_annot_file'.format(phase)])  # reader.COCOADataset(config[train_annot_file])  reader.COCOADataset(config[val_annot_file])
        elif self.dataset == 'Mapillary':
            self.data_reader = reader.MapillaryDataset(
                config['{}_root'.format(phase)], config['{}_annot_file'.format(phase)])
        else:
            self.data_reader = reader.KINSLVISDataset(
                self.dataset, config['{}_annot_file'.format(phase)])
        if config['load_rgb']:
            self.img_transform = transforms.Compose([
                transforms.Normalize(config['data_mean'], config['data_std'])
            ])
        self.eraser_setter = utils.EraserSetter(config['eraser_setter'])  # 只有当两个物体之间的重叠比例大于0.4且小于1.0时/只有当物体被切割的比例大于0.001且小于0.9时，才会应用橡皮擦效果
        self.sz = config['input_size']
        self.eraser_front_prob = config['eraser_front_prob']  # 有80%的概率使用橡皮擦对样本进行擦除，从而增加数据集的多样性，提高模型的泛化能力
        self.phase = phase

        self.config = config

        self.memcached = config.get('memcached', False)
        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)  # 分布式缓存

    def __len__(self):
        return self.data_reader.get_instance_length()

    def _init_memcached(self):
        if not self.initialized:
            assert self.memcached_client is not None, "Please specify the path of your memcached_client"
            server_list_config_file = "{}/server_list.conf".format(self.memcached_client)
            client_config_file = "{}/client.conf".format(self.memcached_client)
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _load_image(self, fn):
        if self.memcached:
            try:
                img_value = mc.pyvector()
                self.mclient.Get(fn, img_value)
                img_value_str = mc.ConvertBuffer(img_value)
                img = utils.pil_loader(img_value_str)
            except:
                print('Read image failed ({})'.format(fn))
                raise Exception("Exit")
            else:
                return img
        else:
            return Image.open(fn).convert('RGB')

    def _get_inst(self, idx, load_rgb=False, randshift=False):
        modal, bbox, category, imgfn, _ = self.data_reader.get_instance(idx)  # imgfn:图片名称

        # 模态掩码图（未截取时）
        temp = modal
        img_pil = Image.fromarray(np.uint8(temp))
        plt.imshow(img_pil)
        plt.show()

        centerx = bbox[0] + bbox[2] / 2.  # 中心点x坐标
        centery = bbox[1] + bbox[3] / 2.  # 中心点y坐标
        size = max([np.sqrt(bbox[2] * bbox[3] * self.config['enlarge_box']), bbox[2] * 1.1, bbox[3] * 1.1])  # 三个值求最大值（长*1.1，宽*1.1）
        if size < 5 or np.all(modal == 0):  # 不过这里
            return self._get_inst(
                np.random.choice(len(self)), load_rgb=load_rgb, randshift=randshift)

        # shift & scale aug 扩大或缩小size
        if self.phase == 'train':
            if randshift:  # 不过
                centerx += np.random.uniform(*self.config['base_aug']['shift']) * size  # np.random.uniform：均匀分布
                centery += np.random.uniform(*self.config['base_aug']['shift']) * size
            size /= np.random.uniform(*self.config['base_aug']['scale'])  # 0.8,1.2

        # crop
        new_bbox = [int(centerx - size / 2.), int(centery - size / 2.), int(size), int(size)]  # 新的边界框
        modal = cv2.resize(utils.crop_padding(modal, new_bbox, pad_value=(0,)),  # 按新的box截取出modal
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        # flip 翻转
        if self.config['base_aug']['flip'] and np.random.rand() > 0.5:
            flip = True
            modal = modal[:, ::-1]  # 沿着第二个维度（列）进行反转
        else:
            flip = False

        if load_rgb:
            rgb = np.array(self._load_image(os.path.join(
                self.config['{}_image_root'.format(self.phase)], imgfn)))  # uint8 加载rgb图片

            # 显示图像
            img_pil = Image.fromarray(np.uint8(rgb))
            plt.imshow(img_pil)
            plt.show()
            print("++++++++++++++++++++++++")

            rgb = cv2.resize(utils.crop_padding(rgb, new_bbox, pad_value=(0,0,0)), # 图像缩放，(428,640,3)=>(256,256,3)
                (self.sz, self.sz), interpolation=cv2.INTER_CUBIC)  # cv2.INTER_CUBIC（4x4像素邻域的双三次插值）
            if flip:
                rgb = rgb[:, ::-1, :]
            rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)  # (256,256,3)=>(3,256,256)
            # rgb = self.img_transform(rgb)  # CHW
            rgb = None  # hzk添加

        if load_rgb:
            return modal, category, rgb
        else:
            return modal, category, None

    def __getitem__(self, idx):
        if self.memcached:
            self._init_memcached()
        randidx = np.random.choice(len(self))  # 用于从给定的数组或者列表中随机选择出一个元素，这个值每次运行时第一次都是960，第二次一直都是xxx，因为DistributedGivenIterationSampler中设置了np.random.seed(0)
        modal, category, rgb = self._get_inst(  # 获取模态掩码、类型、原图
            # idx, load_rgb=self.config['load_rgb'], randshift=True) # modal, uint8 {0, 1}
            idx, load_rgb=True, randshift=True)

        if not self.config.get('use_category', True):  # 不过这里
            category = 1
        eraser, _, _ = self._get_inst(randidx, load_rgb=False, randshift=False)  # 随机获取（遮挡）对象掩码
        eraser = self.eraser_setter(modal, eraser)  # uint8 {0, 1}

        # erase
        erased_modal = modal.copy().astype(np.float32)  # 复制modal
        if np.random.rand() < self.eraser_front_prob:  # 如果随机数小于0.8
            erased_modal[eraser == 1] = 0  # eraser above modal 关联对象掩码在模态掩码之上，将eraser为1的位置全部变成0
        else:  # 否则
            eraser[modal == 1] = 0  # eraser below modal 关联对象掩码在模态掩码之上，将modal为1的位置全部变成0
        erased_modal = erased_modal * category

        # shrink eraser
        max_shrink_pix = self.config.get('max_eraser_shrink', 0)  # 为0
        if max_shrink_pix > 0:  # 不过
            shrink_pix = np.random.choice(np.arange(max_shrink_pix + 1))
            if shrink_pix > 0:
                shrink_kernel = shrink_pix * 2 + 1
                eraser = 1 - cv2.dilate(
                    1 - eraser, np.ones((shrink_kernel, shrink_kernel), dtype=np.uint8),
                    iterations=1)
        eraser_tensor = torch.from_numpy(eraser.astype(np.float32)).unsqueeze(0)  # 1HW，from_numpy:将NumPy数组转换为PyTorch的张量 astype(np.float32)：从默认类型转换成单精度浮点数格式 unsqueeze：在张量的指定维度上增加一个新轴，(256,256)=>(1,256,256)
        # erase rgb
        if rgb is not None:
            rgb = rgb * (1 - eraser_tensor)
        else:
            rgb = torch.zeros((3, self.sz, self.sz), dtype=torch.float32)  # 3HW
        erased_modal_tensor = torch.from_numpy(
            erased_modal.astype(np.float32)).unsqueeze(0)  # 1HW (256,256)=>(1,256,256)，
        target = torch.from_numpy(modal.astype(np.int))  # HW (256,256),将model和eraser折叠后生成target

        # temp = erased_modal_tensor[0]
        # img_pil = Image.fromarray(np.uint8(temp))
        # plt.imshow(img_pil)
        # plt.show()
        #
        # temp = eraser_tensor[0]
        # img_pil = Image.fromarray(np.uint8(temp))
        # plt.imshow(img_pil)
        # plt.show()
        #
        # temp = target
        # img_pil = Image.fromarray(np.uint8(temp))
        # plt.imshow(img_pil)
        # plt.show()

        return rgb, erased_modal_tensor, eraser_tensor, target  #
