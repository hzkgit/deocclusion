import torch
import pycocotools.mask as mask_util
import numpy as np


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


if __name__ == '__main__':
    # print(222)
    # device = torch.device("cuda:0")
    # print(device)
    # print(torch.cuda.get_device_name(0))
    # print(torch.rand(3, 3).cuda())
    rle = code()
    code2(rle)
    code3()
