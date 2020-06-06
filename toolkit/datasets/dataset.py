# -*- encoding: utf-8 -*-

class Dataset(object):
    """数据集父类
    """

    def __init__(self, name, dataset_root):
        """初始化
        Args:
            name:数据集名称
            dataset_root:数据集根目录
        """
        self.name = name
        self.dataset_root = dataset_root
        self.videos = None  # 视频，为字典格式 { "序列名" : 对应的Video对象 }

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]

    def __len__(self):
        return len(self.videos)

    def __iter__(self):
        keys = sorted(list(self.videos.keys()))
        for key in keys:
            yield self.videos[key]

    def set_tracker(self, path, tracker_names):
        """设置tracker
        Args:
            path: tracker输出结果的路径
            tracker_names:
        """
        self.tracker_path = path
        self.tracker_names = tracker_names
