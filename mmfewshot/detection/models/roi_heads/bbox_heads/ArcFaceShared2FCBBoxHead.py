# 文件: mmdet/models/roi_heads/bbox_heads/arcface_shared_2fc_bbox_head.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.models.utils import build_linear_layer
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead


@HEADS.register_module()
class ArcFaceShared2FCBBoxHead(Shared2FCBBoxHead):
    """ArcFace-style BBox head.
    Output: normalized cosine logits (no margin, no scale)
    Loss: MarginArcFaceLoss handles margin and scale.
    """

    def __init__(self,
                 *args,
                 # s=64,
                 num_classes=None,
                 **kwargs):
        super(ArcFaceShared2FCBBoxHead, self).__init__(*args, **kwargs)
        assert num_classes is not None, "num_classes must be provided"
        self.num_classes = num_classes
        # self.s = s

        in_channels = self.fc_out_channels

        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 * self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        # ArcFace weight
        self.weight = nn.Parameter(torch.randn(cls_channels, in_channels))
        nn.init.xavier_uniform_(self.weight)

        # 禁止原fc_cls创建多余参数，避免DDP报错
        self.fc_cls = nn.Identity()

    def forward(self, x, return_fc_feat=False):
        """Forward with ArcFace-style classification."""
        # flatten and shared FCs
        x = x.flatten(1)
        for fc in self.shared_fcs:
            x = self.relu(fc(x))

        cls_feat = x
        reg_feat = self.fc_reg(x)

        # compute normalized cosine similarity
        feat_norm = F.normalize(cls_feat, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(feat_norm, weight_norm)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        # 不乘 self.s，让 MarginArcFaceLoss 自行处理
        if return_fc_feat:
            return cos_theta, reg_feat, cls_feat
        else:
            return cos_theta, reg_feat

    def get_targets(self, *args, **kwargs):
        return super().get_targets(*args, **kwargs)

