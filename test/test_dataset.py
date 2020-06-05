# -*- encoding: utf-8 -*-

"""
@Author : Soarkey
@Date   : 2020/6/5
@Desc   : test_dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import cv2
from tqdm import tqdm

from toolkit.datasets import DatasetFactory
from toolkit.datasets.dataset import Dataset


class TestDataFactory(unittest.TestCase):
    def test_dataset(self):
        """测试数据集类
        """
        dataset = Dataset("OTB100", "/root/OTB100")
        dataset.videos = {"Basketball":"video 1 object", "Biker":"video 2 object"}
        print(len(dataset))
        print(dataset[0])
        print(dataset["Basketball"])

        i = iter(dataset)
        print(next(i))
        print(next(i))

    def test_data_factory(self):
        dataset = DatasetFactory.create_dataset(
            name="OTB100",
            dataset_root="D:/datasets/otb-100-Copy/",
            load_img=False
        )
        video = dataset[0]
        video.start_frame = 1
        video.end_frame = len(video) - 1
        video.load_img()
        video.show()

    def test_view_ground_truth(self):
        """测试显示OTB100所有数据的真值
        """
        dataset = DatasetFactory.create_dataset(
            name="OTB100",
            dataset_root="D:/datasets/otb-100-Copy/",
            load_img=False
        )
        for video in tqdm(dataset):
            for idx, (frame, gt_bbox) in enumerate(video):
                frame = cv2.rectangle(frame,
                                      (gt_bbox[0], gt_bbox[1]),
                                      (gt_bbox[2] + gt_bbox[0], gt_bbox[3] + gt_bbox[1]),
                                      (0, 0, 255),
                                      2)
                # 只针对matplotlib可视化需要
                # if len(frame.shape) == 2:
                #     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                # else:
                #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                cv2.imshow(video.name, frame)
                cv2.waitKey(10)
            cv2.destroyWindow(video.name)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()
