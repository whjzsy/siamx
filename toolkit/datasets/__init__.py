from .otb import OTBDataset


class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """构建数据集
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: whether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'OTB' in name:
            dataset = OTBDataset(**kwargs)
        # elif 'LaSOT' == name:
        #     dataset = LaSOTDataset(**kwargs)
        # elif 'UAV' in name:
        #     dataset = UAVDataset(**kwargs)
        # elif 'NFS' in name:
        #     dataset = NFSDataset(**kwargs)
        # elif 'VOT2018' == name or 'VOT2016' == name or 'VOT2019' == name:
        #     dataset = VOTDataset(**kwargs)
        # elif 'VOT2018-LT' == name:
        #     dataset = VOTLTDataset(**kwargs)
        # elif 'TrackingNet' == name:
        #     dataset = TrackingNetDataset(**kwargs)
        # elif 'GOT-10k' == name:
        #     dataset = GOT10kDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

