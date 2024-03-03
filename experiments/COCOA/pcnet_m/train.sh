#!/bin/bash
# $0表示当前执行脚本的文件名，不包括扩展名 通过dirname $0，我们可以获取这个路径的目录名称，也就是脚本所在的目录
# --nproc_per_node=8 表示每个节点上有8个进程在运行，这些进程可以使用不同的GPU进行计算
# python -m torch.distributed.launch 是一个Python命令行选项，用于启动PyTorch的分布式训练。
# --launcher pytorch 选项指定使用PyTorch的分布式扩展程序作为启动器。
work_path=$(dirname $0)
python -m torch.distributed.launch --nproc_per_node=8 main.py \
    --config $work_path/config.yaml --launcher pytorch
