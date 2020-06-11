# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger('global')

# pytorch分布式训练相关
# 可参考 https://zhuanlan.zhihu.com/p/76638962
parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    pass


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
