# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os
import sys

import cv2
import numpy as np
from torch.utils.data import Dataset

from siamban.core.config import cfg
from siamban.datasets.augmentation import Augmentation
from siamban.datasets.point_target import PointTarget
from siamban.utils.bbox import center2corner, Center

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


# 单个训练数据集
class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        """初始化单个训练数据集
        Args:
            name: 数据集名称
            root: 数据路径
            anno: 标注label
            frame_range:
            num_use: 采样数
            start_idx: 在所有训练数据集里所属的起始索引
        """
        cur_path = os.path.dirname(os.path.realpath(__file__))
        # 数据集名称
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        # 采样的帧范围
        self.frame_range = frame_range
        # 从该数据集中采样的数量
        self.num_use = num_use
        self.start_idx = start_idx

        # 加载训练数据集meta data
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
        meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                                  filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        # 总样本数
        self.num = len(self.labels)
        # -1表示使用所有的样本
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        self.path_format = '{}.{}.{}.jpg'

        logger.info("{} loaded".format(self.name))

        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def __len__(self):
        return self.num

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def log(self):
        """输出本数据集的加载情况
        {数据集名称} start-index {起始位置索引} select [{采样数}/{总样本数}] path_format {图片的路径}
        """
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def get_random_target(self, index=-1):
        """获取一个target

        Args:
            index: 索引下标，默认-1为随机采样

        Returns:

        """
        if index == -1:
            index = np.random.randint(0, self.num)

        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def get_image_anno(self, video, track, frame):
        """得到图像路径和对应的标注
        # Todo ???
        Args:
            video:
            track:
            frame:

        Returns:
            图像路径, 图像的标注
        """
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        """得到正样本对

        Args:
            index: 索引

        Returns:
            template的image和anno, search的image和anno
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))

        # 在template之后的frame_range帧中采样search
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]

        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)

        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame)




# 这里继承了torch.utils.data.Dataset
# BAN的训练数据集类
class BANDataset(Dataset):
    def __init__(self):
        super(BANDataset, self).__init__()
        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
                       cfg.POINT.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create point target
        self.point_target = PointTarget()
        # create sub dataset
        # 存放了所有的训练数据集
        self.all_dataset = []

        # 每个训练数据集的起始索引
        start = 0
        # 所有数据集样本的总数
        self.num = 0

        # 加载所有数据集
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                name,
                subdata_cfg.ROOT,
                subdata_cfg.ANNO,
                subdata_cfg.FRAME_RANGE,
                subdata_cfg.NUM_USE,
                start
            )

            sub_dataset.log()

            start += sub_dataset.num
            self.num += sub_dataset.num_use
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )
        # 每个Epoch使用的视频序列数
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        # 采样总数为epoch * num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        while len(pick) < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p

        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))

        return pick[:self.num]

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        # 是否进行灰度增强
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        # 是否为采样负样本
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            # 负采样, 从对应数据集序列中随机采样template区域,
            # 而从其他的数据集视频序列中采样search区域
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            # 正采样, 从同一个视频序列中采样template/search序列对
            template, search = dataset.get_positive_pair(index)

        # template[0]为图像路径, template[1]为标注
        # 加载图像
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        # get labels
        # cls为类别置信度, delta为bbox偏移量
        cls, delta = self.point_target(bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        # cv2读入图片为H*W*C, 这里使用transpose调整为C*H*W
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
            'template': template,
            'search': search,
            'label_cls': cls,
            'label_loc': delta,
            'bbox': np.array(bbox)
        }

    def _find_dataset(self, index):
        """根据索引找到所属子数据集
        Args:
            index: 索引

        Returns: 子数据集对象, 在该子数据集中的索引
        """
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        """获得bbox
        Args:
            image:图像 (已经读取进内存的图像)
            shape:图像标注

        Returns: bbox坐标 (x1, y1, x2, y2)
        """
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        # Todo ???
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox
