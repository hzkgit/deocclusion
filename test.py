import copy

import torch
import torch.nn as nn
import pycocotools.mask as mask_util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
import pyperclip
import json
import pycocotools.mask as maskUtils
from PIL import Image
import utils
import torch
from torchvision import transforms
import random
import time
import torchvision.utils as vutils
import torch.nn.functional as F


# 将mask转为rle格式（压缩数据）
def code():
    mask = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 1, 0], [0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])
    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    print(rle)
    print(mask != 1)
    mask = (mask == 1)
    print(mask)
    print(~mask)
    return rle


# 将rle转回mask格式（还原）
def code2(rle):
    modal = mask_util.decode(rle).squeeze()
    print(modal)
    print(modal.shape)


def code3():
    a = np.array([True, False])
    print(np.all(a))  # True


def code4():
    width = 8
    height = 6
    rows = 4
    result = 16. / width * height * rows
    print(result)
    print(np.random.random((1, 3)))
    print(np.random.random((1, 3)) * 0.6 + 0.4)


def code5():
    N = 5
    r = [[0.5, 0.5], [0.6, 0.6], [0.7, 0.8], [0.5, 0.8], [0.4, 0.9]]
    polygon = Polygon(r, True)
    fig, ax = plt.subplots()
    ax.add_patch(polygon)
    plt.show()


def code6():
    # 读取图片
    img = cv2.imread('./demos/demo_data/COCOA/4.jpg', 0)
    # 创建内核
    kernel = np.ones((4, 4), np.uint8)
    # 进行膨胀操作
    img_dilate = cv2.dilate(img, kernel, iterations=1)
    # 显示原图和膨胀后的图像
    cv2.imshow("Original", img)
    cv2.imshow("Dilate", img_dilate)
    cv2.waitKey(0)


def code7():
    # 生成一个数组
    arr = np.array([True, False, True, True, False])
    # 使用np.where函数查找True值的索引
    index = np.where(arr)
    # 输出结果
    print(index)


def code8():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ancestors = np.array([0, 1])
    # 计算和并判断是否大于0
    print(data[ancestors, ...])
    print(data[ancestors, ...].sum(axis=0))
    result = data[ancestors, ...].sum(axis=0) > 0
    print(result)


def code9():
    a = torch.Tensor([1, 2])
    print(a.shape)  # 输出：(2,)
    # 在第一个维度增加一个维度
    a = a.unsqueeze(0)
    print(a.shape)  # 输出：(1, 2)
    print(a)


def code10():
    output = torch.randn(5, 5)
    output_softmax = nn.functional.softmax(output, dim=1)
    print(output)
    print(output_softmax)
    output_softmax2 = nn.functional.softmax(output, dim=0)
    print(output_softmax2)


def code11():
    tens = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    arr = tens.numpy()
    print(arr)


def code12():
    ret_modal, ret_bboxes, ret_category, ret_amodal = [], [], [], []
    img_idx = 4
    image_fn = 'demos/demo_data/COCOA/{}.jpg'.format(img_idx)
    fn = 'demos/demo_data/COCOA/{}.json'.format(img_idx)
    img = Image.open(image_fn)
    h, w = img.height, img.width
    with open(fn, 'r') as f:
        ann = json.load(f)
    for reg in ann['regions']:
        if 'visible_mask' in reg.keys():
            # rle = [reg['visible_mask']]
            rle = [reg['invisible_mask']]
            print(rle[0]["counts"])
            print("\n")
            test = maskUtils.decode(rle).squeeze()  # ndarray(640,480)
            s = maskUtils.encode(test)
            print(s["counts"].decode('utf-8'))
        else:
            rles = maskUtils.frPyObjects([reg['segmentation']], h, w)
            rle = maskUtils.merge(rles)
        modal = maskUtils.decode(rle).squeeze()
        if np.all(modal != 1):
            amodal = maskUtils.decode(maskUtils.merge(maskUtils.frPyObjects([reg['segmentation']], h, w)))
            bbox = utils.mask_to_bbox(amodal)
        else:
            bbox = utils.mask_to_bbox(modal)
        ret_modal.append(modal)
        ret_bboxes.append(bbox)
        ret_category.append(1)
        amodal = maskUtils.decode(maskUtils.merge(maskUtils.frPyObjects([reg['segmentation']], h, w)))
        ret_amodal.append(amodal)


def code13():
    path = 'demos/demo_data/AHP/human_modal.png'
    # path = 'demos/demo_data/AHP/human_amodal.png'
    # path = 'demos/demo_data/AHP/human_invisible.png'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    modal = img.clip(max=1)
    mask_merge = np.asfortranarray(modal)  # 不加这句话会报ndarray is not Fortran contiguous
    s = maskUtils.encode(mask_merge)
    ss = s["counts"].decode('utf-8')
    print(ss)


def code14():
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    img = torch.randn(3, 128, 64)
    pic = toPIL(img)
    pic.show()


def code15():
    path = 'demos/demo_data/AHP/human.jpg'
    image = Image.open(path)
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).float()
    temp = image_tensor.numpy()
    img_pil = Image.fromarray(np.uint8(temp))
    plt.imshow(img_pil)
    plt.show()


def code16():
    tens = torch.tensor([[1, 2, 3], [4, 5, 6]]).numpy()
    tens2 = torch.tensor([[0, 0, 0], [0, 0, 0]]).numpy()
    print(np.all(tens == 0))
    print(np.all(tens2 == 0))


def code17():
    arr = np.array([10, 20, 30, 40, 50])
    # p = [0.2, 0.3, 0.2, 0.1, 0.2]  # 第1个元素被选中的概率为20%，第2个元素被选中的概率为30%，以此类推
    # result = np.random.choice(arr, size=3, replace=False, p=p)
    # print(result)
    result = np.random.choice(arr)
    print(result)


def code18():
    # 创建一个形状为(4, 4)的二维数组
    target = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]],
                       [[2, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]],
                       [[3, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]],
                       [[4, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [9, 10, 11, 12]]])
    # 创建一个形状为(4, 4)的布尔型数组，True值对应的是需要保留的元素
    mask = np.array([[[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]],
                     [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]],
                     [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]],
                     [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]]])
    test1 = torch.randn(4, 256, 256).byte()
    test2 = torch.randn(4, 256, 256).byte()
    # 使用布尔索引从target中筛选出在mask中被标记为True的元素
    target_inmask = test1[test2]  # 使用test2作为索引，从test1中选择出一些元素,这行代码会将test1中通过test2选择的元素组成一个新的张量
    print(target_inmask)  # 输出: array([[1, 3], [6, 8]])
    print(target[0, 1, 1])  # 6
    print(target[[0, 1, 1]])  # [6 7]
    print("========================================")
    print(target[mask])


def code19():
    # a = np.arange(64).astype(int).reshape(4, 4, 4)
    # b = np.random.randint(2, size=(4, 4, 4))
    # print(a)
    # print("============================================")
    # print(b)
    # print("============================================")
    # print(a[b])
    # print("============================================")
    test1 = torch.randint(0, 3, size=(2, 4, 4))
    print(test1)
    test2 = torch.randint(0, 3, size=(2, 4, 4))
    # print(test2)
    test2 = test2.byte()  # 浮点数转换位字节(范围0~255)
    print(test2)
    print("============================================")
    print(test1[test2])


def code20():
    test1 = torch.randint(0, 256, size=(2, 4, 4))
    test2 = torch.randint(0, 2, size=(2, 4, 4)).byte()
    print(test1)
    print(test2)
    print(~test2)
    print(test1[test2])  # test2与test1对应位置元素，如果test2对应位置>0则test1对应元素保留，否则test1对应元素去掉
    print(test1[~test2])


def code21():
    # 获取三维tensor
    path = 'experiments/COCOA/pcnet_m/images/on_val_16000.png'
    image = Image.open(path)
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).float()

    # 转为图像显示
    temp = image_tensor.numpy() * 100
    img_pil = Image.fromarray(np.uint8(temp))
    plt.imshow(img_pil)
    plt.show()


def code22():
    # 获取三维tensor
    path = 'demos/demo_data/AHP/human.jpg'
    image = Image.open(path)
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).float()

    # 转为图像显示
    temp = image_tensor.numpy()
    img_pil = Image.fromarray(np.uint8(temp))  # (480,640,3)
    plt.imshow(img_pil)
    plt.show()


def code23():
    btime_rec = utils.AverageMeter()  # 10为最大长度
    for i in range(0, 7):
        btime_rec.update(float(i))
    print(btime_rec.count)  # 当不指定最大长度时有效，有多少个值
    print(btime_rec.sum)  # 当不指定最大长度时有效，所有值的和
    print(btime_rec.val)  # 最近一次输入的值
    print(btime_rec.avg)  # 所有值的平均数


def code24():
    btime_rec = utils.AverageMeter(10)  # 10为最大长度
    for i in range(0, 15):
        btime_rec.update(float(i))
    print(btime_rec.length)  # 当指定最大长度时有效，表示最大长度
    print(btime_rec.history)  # 当指定最大长度时有效，表示当前有哪些元素
    print(btime_rec.val)  # 最近一次输入的值
    print(btime_rec.avg)  # 所有值的平均数


def code25():
    # 将多张图像放到一张图像中并列显示
    path = 'demos/demo_data/AHP/human.jpg'
    image = Image.open(path)
    image_np = np.array(image)
    image_tensor = torch.from_numpy(image_np).float()  # .unsqueeze(0)
    path2 = 'demos/demo_data/AHP/human_modal.png'
    image2 = Image.open(path2)
    image_np2 = np.array(image2)
    image_tensor2 = torch.from_numpy(image_np2).float().unsqueeze(2).repeat(1, 1, 3)
    images = [image_tensor, image_tensor2, image_tensor, image_tensor2]
    images_tensor = torch.stack(images).permute(0, 3, 1, 2)  # (4,3,480,640)
    # make_grid函数的作用是将多个图像合并成一个数组，并在一个图像中显示出来
    # grid = vutils.make_grid(temp,  # 输入的张量，一般为大小为 (B, C, H, W) 的四维张量，其中 B 是批次大小，C 是通道数，H 和 W 分别是每张图像的高度和宽度。
    #                         nrow=2,  # 每行显示的图像数量，默认为 8
    #                         padding=2,  # 每个图像之间的像素填充，默认为 2
    #                         normalize=False,  # 是否进行归一化，默认为 False。如果设置为 True，则将图像像素值归一化到 [0, 1] 范围。
    #                         # range=(0, 255),  # 将图像像素值缩放到指定范围，默认为 None。如果 range 给定为一个元组 (min, max)，则将像素值缩放到该范围内。
    #                         scale_each=False,  # 是否对每个图像独立进行像素缩放，默认为 False。如果设置为 True，则将每个图像的像素值独立缩放
    #                         pad_value=0)  # 填充像素的值，默认为 0
    grid = vutils.make_grid(images_tensor,
                            nrow=2,
                            normalize=True,
                            range=(0, 255),
                            scale_each=False)
    img_pil = Image.fromarray(np.uint8(grid.permute(1, 2, 0)) * 255)
    plt.imshow(img_pil)
    plt.show()
    # vutils.save_image(grid, 'grid_image.png')
    # image_grid = Image.fromarray(grid.permute(1, 2, 0).numpy())  # 将[1, 28, 28]转换为[28, 28, 1]
    # image_grid.show()


def code26():
    test = torch.randint(1, 5, size=(2, 4, 4)).float()
    test2 = test.argmax(dim=1, keepdim=True)  # keepdim=True 保持输出张量的维度不变
    test3 = test.argmax(dim=0, keepdim=True)  # keepdim=True 保持输出张量的维度不变
    test4 = test.argmax(dim=2, keepdim=True)  # keepdim=True 保持输出张量的维度不变
    print(test)  # (2,4,4)
    print(test2)  # (2,1,4)
    print(test3)  # (1,4,4)
    print(test4)  # (2,4,1)
    print(123)

    # arr = torch.tensor([2.0, 3.5, 0.5, 4.7])
    # out = nn.functional.normalize(test, p=1, dim=3)
    # print(out)


def code27():
    # test1 = torch.randint(0, 5, size=(26972, 2)).float() # 26972个样本，2种类型
    # test2 = torch.randint(0, 1, size=(26972,)).long()  # 26972个样本对应的正确类型
    # print(test1)
    # print(test2)

    test1 = torch.rand((4, 2)).float()  # 4个样本，2种类型
    test1_softmax = F.softmax(test1)
    test1_log = torch.log(test1_softmax)
    test2 = torch.randint(0, 2, size=(4,)).long()  # 4个样本对应的正确类型
    print(test1)
    print(test1_softmax)
    print(test1_log)
    print(test2)

    loss = nn.functional.cross_entropy(  # L1
        test1, test2, size_average=False)  # size_average是否计算损失值的平均
    print(loss)


def code28():
    x = np.array([[1, 2, 3, 4, 5],  # 共三3样本，有5个类别
                  [1, 2, 3, 4, 5],
                  [1, 2, 3, 4, 5]]).astype(np.float32)
    y = np.array([1, 1, 0])  # 这3个样本的标签分别是1,1,0即两个是第2类，一个是第1类,相当于
    # [[0,1,0,0,0]
    #  [0,1,0,0,0]
    #  [1,0,0,0,0]]
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).long()
    soft_out = F.softmax(x, dim=1)  # 给每个样本的pred向量做指数归一化---softmax
    log_soft_out = torch.log(soft_out)  # 将上面得到的归一化的向量再point-wise取对数
    loss = F.nll_loss(log_soft_out, y)  # 将归一化且取对数后的张量根据标签求和，实际就是计算loss的过程
    """
    这里的loss计算式根据batch_size归一化后的，即是一个batch的平均单样本的损失，迭代一次模型对一个样本平均损失。
    在多个epoch训练时，还会求每个epoch内的总损失，用于衡量epoch之间模型性能的提升。
    """
    print(soft_out)
    print(log_soft_out)
    print(loss)
    loss = F.cross_entropy(x, y)
    print(loss)


def code29():
    test1 = torch.rand((4, 256, 256, 3)) * 255  # 彩色
    # test1 = torch.rand((256, 256, 1)).repeat(1, 1, 3)*255  # 彩色

    # # 转为图像显示(但仅限一张图片)
    # temp = test1.numpy()
    # img_pil = Image.fromarray(np.uint8(temp))
    # plt.imshow(img_pil)
    # plt.show()

    # 显示多张图片
    test2 = test1.permute(0, 3, 1, 2)  # (4, 3, 256, 256)
    grid = vutils.make_grid(test2,
                            nrow=2,
                            normalize=True,
                            range=(0, 255),
                            scale_each=False)
    img_pil = Image.fromarray(np.uint8(grid.permute(1, 2, 0) * 255))  # * 255
    plt.imshow(img_pil)
    plt.show()


def code30():
    num = random.uniform(2, 10)
    test1 = torch.rand((4, 2, 256, 256)) * num

    # 将元素值全部归到（0,1）之间
    temp = F.normalize(test1, p=2, dim=None)


    t1 = temp[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
    t2 = temp[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)

    images_tensor = torch.cat((t1, t2), dim=0)

    grid = vutils.make_grid(images_tensor,
                            nrow=4,
                            normalize=True,
                            range=(0, 255),
                            scale_each=False)

    img_pil = Image.fromarray(np.uint8(grid.permute(1, 2, 0) * 255))
    plt.imshow(img_pil)
    plt.show()


def code31():
    test1 = torch.rand((4, 2, 256, 256))
    test2 = copy.deepcopy(test1)
    test1 = test1.transpose(1, 2).transpose(2, 3).contiguous()
    print(123)


def code32():
    np.random.seed(0)
    index = np.random.choice(22163)
    print(index)
    index = np.random.choice(22163)
    print(index)


if __name__ == '__main__':
    # print(222)
    # device = torch.device("cuda:0")
    # print(device)
    # print(torch.cuda.get_device_name(0))
    # print(torch.rand(3, 3).cuda())
    # rle = code()
    # code2(rle)
    # code3()
    # code4()
    # code5()
    # code6()
    # code7()
    # code8()
    # code9()
    # code10()
    # code11()
    # code12()
    # code13()
    # code14()
    # code15()
    # code16()
    # code17()
    # code18()
    # code19()
    # code20()
    # code21()
    # code22()
    # code23()
    # code24()
    # code25()
    # code26()
    # code27()
    # print("============================")
    # code28()
    # code29()
    # code30()
    # code31()
    code32()
