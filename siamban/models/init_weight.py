
import torch.nn as nn


def init_weights(model):
    for m in model.modules():
        # Conv2d层使用凯明初始化
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out',
                                    nonlinearity='relu')
        # BatchNorm2d层固定值初始化
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
