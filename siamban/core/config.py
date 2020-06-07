# -*- encoding: utf-8 -*-
"""
默认配置管理
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

cfg = CN()
cfg.META_ARC = "siamban_r50_1234"

# 开启CUDA加速
cfg.CUDA = True

# ------------------------------------------------------------------------ #
# 训练配置
# ------------------------------------------------------------------------ #
cfg.TRAIN = CN()

# 负样本取样数
cfg.TRAIN.NEG_NUM = 16

# 正样本取样数
cfg.TRAIN.POS_NUM = 16

# Todo ???
cfg.TRAIN.TOTAL_NUM = 64

cfg.TRAIN.EXEMPLAR_SIZE = 127
cfg.TRAIN.SEARCH_SIZE = 255

# Todo ???
cfg.TRAIN.BASE_SIZE = 8

# Todo ???
cfg.TRAIN.OUTPUT_SIZE = 25

# 加载模型参数
cfg.TRAIN.RESUME = ''

# 加载预训练模型(本项目中一般指ResNet50)
cfg.TRAIN.PRETRAINED = ''

# 训练日志存放路径
cfg.TRAIN.LOG_DIR = './logs'

# 阶段性保存训练模型的路径
cfg.TRAIN.SNAPSHOT_DIR = './snapshot'

cfg.TRAIN.EPOCH = 20
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.NUM_WORKERS = 1
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0001

# Todo 混合系数 ???
cfg.TRAIN.CLS_WEIGHT = 1.0
cfg.TRAIN.LOC_WEIGHT = 1.0

# Todo ???
cfg.TRAIN.PRINT_FREQ = 20
cfg.TRAIN.LOG_GRADS = False

# 梯度截断
cfg.TRAIN.GRAD_CLIP = 10.0

# 基础学习率
cfg.TRAIN.BASE_LR = 0.005

# 变化学习率的参数
cfg.TRAIN.LR = CN()
# 变化类型
cfg.TRAIN.LR.TYPE = 'log'
# new_allowed: 是否允许在合并yaml的时候添加新的键
cfg.TRAIN.LR.KWARGS = CN(new_allowed=True)
# 热启动参数
cfg.TRAIN.LR_WARMUP = CN()
cfg.TRAIN.LR_WARMUP.WARMUP = True
cfg.TRAIN.LR_WARMUP.TYPE = 'step'
cfg.TRAIN.LR_WARMUP.EPOCH = 5
cfg.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# 数据集配置
# ------------------------------------------------------------------------ #
cfg.DATASET = CN(new_allowed=True)

# Augmentation for template
cfg.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703) 
# for detail discussion
cfg.DATASET.TEMPLATE.SHIFT = 4
cfg.DATASET.TEMPLATE.SCALE = 0.05
cfg.DATASET.TEMPLATE.BLUR = 0.0
cfg.DATASET.TEMPLATE.FLIP = 0.0
cfg.DATASET.TEMPLATE.COLOR = 1.0

# Augmentation for search region
cfg.DATASET.SEARCH = CN()
cfg.DATASET.SEARCH.SHIFT = 64
cfg.DATASET.SEARCH.SCALE = 0.18
cfg.DATASET.SEARCH.BLUR = 0.0
cfg.DATASET.SEARCH.FLIP = 0.0
cfg.DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
cfg.DATASET.NEG = 0.2

# Todo ???
# improve tracking performance for otb100
cfg.DATASET.GRAY = 0.0

# 数据集名称
cfg.DATASET.NAMES = ('VID', 'YOUTUBEBB', 'DET', 'COCO', 'GOT10K', 'LASOT')

cfg.DATASET.VID = CN()
cfg.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
cfg.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
cfg.DATASET.VID.FRAME_RANGE = 100
cfg.DATASET.VID.NUM_USE = 100000

cfg.DATASET.YOUTUBEBB = CN()
cfg.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
cfg.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
cfg.DATASET.YOUTUBEBB.FRAME_RANGE = 3
cfg.DATASET.YOUTUBEBB.NUM_USE = 200000

cfg.DATASET.COCO = CN()
cfg.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
cfg.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
cfg.DATASET.COCO.FRAME_RANGE = 1
cfg.DATASET.COCO.NUM_USE = 100000

cfg.DATASET.DET = CN()
cfg.DATASET.DET.ROOT = 'training_dataset/det/crop511'
cfg.DATASET.DET.ANNO = 'training_dataset/det/train.json'
cfg.DATASET.DET.FRAME_RANGE = 1
cfg.DATASET.DET.NUM_USE = 200000

cfg.DATASET.GOT10K = CN()
cfg.DATASET.GOT10K.ROOT = 'training_dataset/got_10k/crop511'
cfg.DATASET.GOT10K.ANNO = 'training_dataset/got_10k/train.json'
cfg.DATASET.GOT10K.FRAME_RANGE = 100
cfg.DATASET.GOT10K.NUM_USE = 200000

cfg.DATASET.LASOT = CN()
cfg.DATASET.LASOT.ROOT = 'training_dataset/lasot/crop511'
cfg.DATASET.LASOT.ANNO = 'training_dataset/lasot/train.json'
cfg.DATASET.LASOT.FRAME_RANGE = 100
cfg.DATASET.LASOT.NUM_USE = 200000

# 每个Epoch的视频序列数
cfg.DATASET.VIDEOS_PER_EPOCH = 1000000

# ------------------------------------------------------------------------ #
# 骨干网络配置
# ------------------------------------------------------------------------ #
cfg.BACKBONE = CN()

# Backbone type, current only support resnet18, 34, 50; alexnet; mobilenet
cfg.BACKBONE.TYPE = 'res50'
cfg.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
cfg.BACKBONE.PRETRAINED = ''

# Train layers
cfg.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Todo ???
# Layer LR
cfg.BACKBONE.LAYERS_LR = 0.1

# Todo ???
# Switch to train layer
cfg.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
cfg.ADJUST = CN()

# Adjust layer
cfg.ADJUST.ADJUST = True
cfg.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
cfg.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# BAN options
# ------------------------------------------------------------------------ #
cfg.BAN = CN()

# Whether to use ban head
cfg.BAN.BAN = False

# BAN type
cfg.BAN.TYPE = 'MultiBAN'
cfg.BAN.KWARGS = CN(new_allowed=True)

# Todo ???
# ------------------------------------------------------------------------ #
# Point options
# ------------------------------------------------------------------------ #
cfg.POINT = CN()

# Point stride
# 采样点的步长
cfg.POINT.STRIDE = 8

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
cfg.TRACK = CN()
cfg.TRACK.TYPE = 'SiamBANTracker'

# Scale penalty
cfg.TRACK.PENALTY_K = 0.14

# Window influence
cfg.TRACK.WINDOW_INFLUENCE = 0.45

# Interpolation learning rate
cfg.TRACK.LR = 0.30

# Exemplar size
cfg.TRACK.EXEMPLAR_SIZE = 127

# Instance size
cfg.TRACK.INSTANCE_SIZE = 255

# Base size
cfg.TRACK.BASE_SIZE = 8

# Context amount
cfg.TRACK.CONTEXT_AMOUNT = 0.5
