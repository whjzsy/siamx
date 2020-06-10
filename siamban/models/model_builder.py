# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.head(self.zf, xf)
        return {
            'cls': cls,
            'loc': loc
        }

    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            # Whether to use ban head
            # contiguous开辟一块新的连续内存块存放cls, 访问速度会更快, 以空间换时间
            # permute, 把B*cls*H*W变为B*H*W*cls
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def get_cls_loss(self, pred, label, select):
        if len(select.size()) == 0 or select.size() == torch.Size([0]):
            return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return F.nll_loss(pred, label)

    def select_cross_entropy_loss(self, pred, label):
        pred = self.log_softmax(pred)
        pred = pred.view(-1, 2)
        label = label.view(-1)
        pos_index = label.data.eq(1).nonzero().squeeze().cuda()
        neg_index = label.data.eq(0).nonzero().squeeze().cuda()
        loss_pos = self.get_cls_loss(pred, label, pos_index)
        loss_neg = self.get_cls_loss(pred, label, neg_index)
        return (loss_pos + loss_neg) / 2

    def weight_l1_loss(self, pred_loc, label_loc, loss_weight):
        if cfg.BAN.BAN:
            diff = (pred_loc - label_loc).abs()
            diff = diff.sum(dim=1)
        else:
            diff = None
        loss = diff * loss_weight
        return loss.sum().div(pred_loc.size()[0])

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        cls, loc = self.head(zf, xf)

        # get loss
        # cls loss with cross entropy loss
        cls_loss = self.select_cross_entropy_loss(cls, label_cls)

        # loc loss with iou loss
        loc_loss = self.select_iou_loss(loc, label_loc, label_cls)

        outputs = {
            'total_loss':
                cfg.TRAIN.CLS_WEIGHT * cls_loss +
                cfg.TRAIN.LOC_WEIGHT * loc_loss,
            'cls_loss': cls_loss,
            'loc_loss': loc_loss
        }

        return outputs
