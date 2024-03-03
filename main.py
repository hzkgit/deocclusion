import multiprocessing as mp  # mp 是Python中的一个内置模块，用于管理多进程编程
import argparse
import os
import yaml
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['HOME'] = "F:\ML\deocclusion\demos"
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '35632'


from utils import dist_init
from trainer import Trainer


def main(args):
    with open(args.config) as f:
        config = yaml.load(f)  # 加载config文件

    for k, v in config.items():
        setattr(args, k, v)  # setattr：将config文件中的key和value设置到args中

    # exp path
    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    # dist init
    # Windows不支持nccl backend，将nccl改成gloo
    if mp.get_start_method(allow_none=True) != 'spawn':  # 检查当前是否正在使用spawn方式启动进程池（spawn方式：子进程不会直接使用主进程的资源，而是重新执行主进程的代码）
        mp.set_start_method('spawn', force=True)  # 将进程池启动方式设置为 spawn，强制所有进程使用 spawn 方式启动
    # dist_init(args.launcher, backend='nccl')  # 使用 NCCL 库实现分布式计算  dist_init 函数的作用是根据传入的参数，初始化一个分布式计算环境，并将其保存在一个全局变量中，以便后续的操作使用。NCCL（Next Generation Communication Library）是一个开源的分布式计算库，提供了一系列用于实现高性能、可扩展的并行计算和通信的函数和工具
    dist_init(args.launcher, backend='gloo')

    # train 传入参数
    trainer = Trainer(args)
    trainer.run()  # 启动训练


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch De-Occlusion.')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--launcher', default='pytorch', type=str)
    parser.add_argument('--load-iter', default=None, type=int)  # 初始值为None
    parser.add_argument('--load-pretrain', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluate-save', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    main(args)
