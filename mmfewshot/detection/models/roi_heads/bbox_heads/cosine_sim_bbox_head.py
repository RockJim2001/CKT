# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple, Optional
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads import ConvFCBBoxHead
from mmdet.models.losses import accuracy
from torch import Tensor, sigmoid

from mmfewshot.detection.models import ArcFaceLoss
from mmfewshot.utils import logger


@HEADS.register_module()
class CosineSimBBoxHead(ConvFCBBoxHead):
    """BBOxHead for `TFA <https://arxiv.org/abs/2003.06957>`_.

    The code is modified from the official implementation
    https://github.com/ucbdrive/few-shot-object-detection/

    Args:
        scale (int): Scaling factor of `cls_score`. Default: 20.
        learnable_scale (bool): Learnable global scaling factor.
            Default: False.
        eps (float): Constant variable to avoid division by zero.
    """

    def __init__(self,
                 scale: int = 20,
                 learnable_scale: bool = False,
                 eps: float = 1e-5,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # override the fc_cls in :obj:`ConvFCBBoxHead`
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # learnable global scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.scale = scale
        self.eps = eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_cls_normalized = x_cls.div(x_norm + self.eps)
        # normalize weight
        with torch.no_grad():
            temp_norm = torch.norm(
                self.fc_cls.weight, p=2,
                dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
            self.fc_cls.weight.div_(temp_norm + self.eps)
        # calculate and scale cls_score
        cls_score = self.scale * self.fc_cls(
            x_cls_normalized) if self.with_cls else None

        return cls_score, bbox_pred


@HEADS.register_module()
class DisCosSimBBoxHead(CosineSimBBoxHead):
    def __init__(self,
                 dis_loss=None,
                 with_weight_decay=False,
                 decay_rate=1.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if dis_loss is not None:
            self.dis_loss = build_loss(copy.deepcopy(dis_loss))
        else:
            self.dis_loss = None

        self.with_weight_decay = with_weight_decay
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = decay_rate

    def set_decay_rate(self, decay_rate: float):
        """Contrast loss weight decay hook will set the `decay_rate` according
        to iterations.

        Args:
            decay_rate (float): Decay rate for weight decay.
        """
        self._decay_rate = decay_rate

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             cos_dis=None,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # not use bbox sampling
            # pos_inds[1024:] = False
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        if self.with_weight_decay:
            decay_rate = self._decay_rate
        else:
            decay_rate = None

        if self.dis_loss is not None and cls_score is not None:
            losses.update(self.dis_loss(cls_score, labels, label_weights, decay_rate))

        return losses


@HEADS.register_module()
class ArcFaceBBoxHead(CosineSimBBoxHead):
    """BBox head with learnable ArcFace-style classification loss.

    This head keeps the cosine normalization of CosineSimBBoxHead,
    but replaces the classification loss with ArcFaceLoss.
    """

    def __init__(self,
                 margin: float = 0.5,
                 s: float = 64,
                 learnable_margin: bool = True,
                 arcface_loss_cfg=dict(
                     type='ArcFaceLoss',
                     s=64.0,
                     margin=0.5,
                     loss_weight=1.0,
                     reduction='mean'),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_classes = kwargs.get("num_classes")

        # ===== 初始化 margin =====
        if learnable_margin:
            self.margin = nn.Parameter(torch.tensor(margin))
            # self.margin = nn.Parameter(torch.cat([
            #     torch.ones(num_classes) * margin,
            #     torch.zeros(1)  # background margin = 0
            # ]))
            # self.margin = nn.Parameter(torch.cat([torch.tanh(torch.ones(num_classes) * m_init), torch.zeros(1)]))
        else:
            self.register_buffer('margin', torch.tensor(margin))

        self.s = s

        # ===== 初始化 ArcFaceLoss =====
        # 可以通过 build_loss 支持 MMDet 配置风格
        self.loss_cls = build_loss(arcface_loss_cfg)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)

        # ====== ArcFace 分类部分 ======
        # 特征与权重归一化
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)

        # 基本 cosine 相似度
        cosine = F.linear(x_norm, w_norm)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # ✅ 在角度空间引入 learnable margin，保持梯度图完整
        theta = torch.acos(cosine)
        cosine_m = torch.cos(theta + self.margin)
        cls_score = self.s * cosine_m  # 缩放输出

        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        """Compute classification and bbox losses using ArcFaceLoss."""

        losses = dict()

        # ===== 分类分支 =====
        if cls_score is not None and cls_score.numel() > 0:
            # avg_factor 可以根据 label_weights 自动计算
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)

            # 调用 ArcFaceLoss
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                weight=label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override,
                margin=self.margin  # 传入可学习 margin
            )

            # 计算分类准确率（可选）
            losses['acc'] = accuracy(cls_score, labels)

        # ===== 回归分支 =====
        # if bbox_pred is not None:
        #     loss_bbox = self.loss_bbox(
        #         bbox_pred,
        #         bbox_targets,
        #         bbox_weights,
        #         reduction_override=reduction_override)
        #     losses['loss_bbox'] = loss_bbox
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses


@HEADS.register_module()
class DualMarginArcFaceBBoxHead(CosineSimBBoxHead):
    """BBox head with separate learnable margins for base and novel classes."""

    def __init__(self,
                 margin_base: float = 0.5,
                 margin_novel: float = 0.3,
                 s: float = 64,
                 base_class_count: int = 60,  # ✅ 需根据任务传入
                 learnable_margin_base: bool = True,
                 learnable_margin_novel: bool = True,
                 arcface_loss_cfg=dict(
                     type='ArcFaceLoss',
                     s=64.0,
                     margin=0.5,
                     loss_weight=1.0,
                     reduction='mean'),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_classes = kwargs.get("num_classes")
        self.base_class_count = base_class_count
        self.s = s

        # ===== 初始化 margin =====
        if learnable_margin_base:
            self.margin_base = nn.Parameter(torch.tensor(margin_base))
        else:
            self.register_buffer('margin_base', torch.tensor(margin_base))
        if learnable_margin_novel:
            self.margin_novel = nn.Parameter(torch.tensor(margin_novel))
        else:
            self.register_buffer('margin_novel', torch.tensor(margin_novel))

        # ===== 初始化 ArcFaceLoss =====
        self.loss_cls = build_loss(arcface_loss_cfg)

    def get_dynamic_margin(self, labels):
        """根据 label 自动选择 base / novel margin"""
        # 构造 margin 向量
        margin = torch.where(
            labels < self.base_class_count,
            self.margin_base,
            self.margin_novel
        )
        return margin

    def forward(self, x: Tensor, labels: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward function with class-dependent margin"""
        # shared & separate 部分同原实现
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls, x_reg = x, x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # ===== ArcFace 分类部分 =====
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cosine)

        # ✅ 动态 margin（只有训练阶段才有 labels）
        if labels is not None:
            dynamic_margin = self.get_dynamic_margin(labels).view(-1, 1)
            cosine_m = torch.cos(theta + dynamic_margin)
        else:
            # 推理阶段使用 base margin（或均值）
            cosine_m = torch.cos(theta + self.margin_base)

        cls_score = self.s * cosine_m
        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        """Compute classification and bbox losses using dual-margin ArcFaceLoss."""
        losses = dict()

        if cls_score is not None and cls_score.numel() > 0:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)

            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                weight=label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override
            )
            losses['acc'] = accuracy(cls_score, labels)

        # ===== 回归分支同原实现 =====
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                        pos_inds, labels[pos_inds]
                    ]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds],
                    bbox_weights[pos_inds],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses


@HEADS.register_module()
class DualMarginArcFaceDisBBoxHead(DualMarginArcFaceBBoxHead):
    """Dual-margin ArcFace head + DisLoss regularization for incremental few-shot detection."""

    def __init__(self,
                 dis_loss_cfg=dict(
                     type='DisLoss',
                     num_classes=80,  # ✅ 需与当前任务类别总数保持一致
                     shot=5,
                     loss_weight=1.0,
                     reduction='sum'),
                 use_dis_loss=True,
                 dis_loss_weight=0.1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dis_loss = use_dis_loss
        self.dis_loss_weight = dis_loss_weight

        if self.use_dis_loss:
            self.loss_dis = build_loss(dis_loss_cfg)

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        """Compute classification, bbox, and distillation-based losses."""
        # === 原始 ArcFace + BBox 回归损失 ===
        losses = super().loss(
            cls_score, bbox_pred, rois, labels,
            label_weights, bbox_targets, bbox_weights,
            reduction_override, **kwargs)

        # === 新增 DisLoss 正则项 ===
        if self.use_dis_loss and cls_score is not None and cls_score.numel() > 0:
            dis_losses = self.loss_dis(
                cls_score=cls_score,
                labels=labels,
                label_weights=label_weights
            )

            # 各子项损失乘权重后加入总loss
            for k, v in dis_losses.items():
                losses[k] = v * self.dis_loss_weight

        return losses


@HEADS.register_module()
class DualMarginArcFaceKDDisBBoxHead(DualMarginArcFaceDisBBoxHead):
    """Dual-margin ArcFace head with Base/Novel dual branches + KD regularization."""

    def __init__(self,
                 base_cpt=None,
                 base_alpha=0.5,
                 loss_kd_weight=0.001,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert base_cpt is not None
        self.base_cpt = base_cpt
        self.base_alpha = base_alpha
        self.loss_kd_weight = loss_kd_weight

        # === 增加双分支全连接层 ===
        self.base_shared_fcs = nn.ModuleList()
        self.novel_shared_fcs = nn.ModuleList()

        for fc in self.shared_fcs:
            in_dim = fc.in_features
            out_dim = fc.out_features
            self.base_shared_fcs.append(nn.Linear(in_dim, out_dim))
            self.novel_shared_fcs.append(nn.Linear(in_dim, out_dim))

        del self.shared_fcs

        self.loss_kd = dict()
        self.relu = nn.ReLU(inplace=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        torch.manual_seed(0)
        processing = True
        # During training, processing the checkpoints
        # During testing, directly load the checkpoints
        for param_name in state_dict:
            if 'novel_shared_fcs' in param_name:
                processing = False
                break
        print('-------processing-----', processing)

        if processing:
            # load base training checkpoint
            num_base_classes = self.num_classes // 4 * 3
            base_weights = torch.load(self.base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']

            for n, p in base_weights.items():
                if 'bbox_head' not in n:
                    state_dict[n] = copy.deepcopy(p)

            # initialize the base branch with the weight of base training
            for n, p in base_weights.items():
                if 'shared_fcs' in n:
                    new_n_base = n.replace('shared_fcs', 'base_shared_fcs')
                    state_dict[new_n_base] = copy.deepcopy(p)
                    new_n_novel = n.replace('shared_fcs', 'novel_shared_fcs')
                    state_dict[new_n_novel] = copy.deepcopy(p)

                if 'fc_cls' in n:
                    state_dict[n] = copy.deepcopy(p)

                if not self.reg_class_agnostic and 'fc_reg' in n:
                    state_dict[n] = copy.deepcopy(p)

            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, x, labels=None):
        """Forward with dual FC branches for base/novel learning."""
        # --- 特征提取 ---
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)

        loss_feature_kd = 0
        alpha = self.base_alpha
        base_x = x
        kd_loss_list = []
        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1 - alpha) * novel_x
            kd_loss_list.append(torch.frobenius_norm(base_x - x, dim=-1))
            x = self.relu(x)
        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)

        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] = kd_loss

        # --- 分类与回归分支 ---
        x_cls, x_reg = x, x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # === ArcFace 分类逻辑保持不变 ===
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)

        if labels is not None:
            dynamic_margin = self.get_dynamic_margin(labels).view(-1, 1)
            cosine_m = torch.cos(theta + dynamic_margin)
        else:
            cosine_m = torch.cos(theta + self.margin_base)

        cls_score = self.s * cosine_m
        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        """Compute classification, bbox, KD and Dis losses."""
        losses = super().loss(
            cls_score, bbox_pred, rois,
            labels, label_weights,
            bbox_targets, bbox_weights,
            reduction_override, **kwargs)

        # === 增加 KD loss ===
        if self.training and hasattr(self, 'loss_kd'):
            losses.update(self.loss_kd)

        return losses


@HEADS.register_module()
class DualMarginArcFaceOrthDisBBoxHead(DualMarginArcFaceDisBBoxHead):
    """Dual-margin ArcFace head with Base/Novel dual branches + Orthogonal regularization."""

    def __init__(self,
                 base_cpt=None,
                 base_alpha=0.5,
                 loss_orth_weight=0.001,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert base_cpt is not None
        self.base_cpt = base_cpt
        self.base_alpha = base_alpha
        self.loss_orth_weight = loss_orth_weight

        # === 增加双分支全连接层 ===
        self.base_shared_fcs = nn.ModuleList()
        self.novel_shared_fcs = nn.ModuleList()

        for fc in self.shared_fcs:
            in_dim = fc.in_features
            out_dim = fc.out_features
            self.base_shared_fcs.append(nn.Linear(in_dim, out_dim))
            self.novel_shared_fcs.append(nn.Linear(in_dim, out_dim))

        del self.shared_fcs

        self.relu = nn.ReLU(inplace=False)
        self.loss_orth = dict()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # === 加载 base 阶段权重 ===
        torch.manual_seed(0)
        processing = True
        for param_name in state_dict:
            if 'novel_shared_fcs' in param_name:
                processing = False
                break
        print('-------processing-----', processing)

        if processing:
            base_weights = torch.load(self.base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']

            for n, p in base_weights.items():
                if 'bbox_head' not in n:
                    state_dict[n] = copy.deepcopy(p)

            for n, p in base_weights.items():
                if 'shared_fcs' in n:
                    new_n_base = n.replace('shared_fcs', 'base_shared_fcs')
                    state_dict[new_n_base] = copy.deepcopy(p)
                    new_n_novel = n.replace('shared_fcs', 'novel_shared_fcs')
                    state_dict[new_n_novel] = copy.deepcopy(p)

                if 'fc_cls' in n or ('fc_reg' in n and not self.reg_class_agnostic):
                    state_dict[n] = copy.deepcopy(p)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys,
                                      unexpected_keys, error_msgs)

    def forward(self, x, labels=None):
        """Forward with orthogonal Base/Novel branches."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)

        alpha = self.base_alpha
        loss_orth_value = 0

        for fc_b, fc_n in zip(self.base_shared_fcs, self.novel_shared_fcs):
            base_x = fc_b(x)
            novel_x = fc_n(x)
            x = alpha * base_x + (1 - alpha) * novel_x

            if self.training:
                # === 计算正交约束 ===
                base_norm = F.normalize(base_x, p=2, dim=1)
                novel_norm = F.normalize(novel_x, p=2, dim=1)
                cosine_sim = torch.sum(base_norm * novel_norm, dim=1)  # cos(angle)
                loss_orth_value += torch.mean(cosine_sim ** 2)  # 惩罚相似方向

            x = self.relu(x)

        if self.training:
            self.loss_orth['loss_orth'] = loss_orth_value * self.loss_orth_weight

        # === 分类分支 ===
        x_cls, x_reg = x, x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # === ArcFace 分类 ===
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)

        if labels is not None:
            dynamic_margin = self.get_dynamic_margin(labels).view(-1, 1)
            cosine_m = torch.cos(theta + dynamic_margin)
        else:
            cosine_m = torch.cos(theta + self.margin_base)

        cls_score = self.s * cosine_m
        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        """Compute classification, bbox, Dis and Orth losses."""
        losses = super().loss(
            cls_score, bbox_pred, rois,
            labels, label_weights,
            bbox_targets, bbox_weights,
            reduction_override, **kwargs)

        if self.training and hasattr(self, 'loss_orth'):
            losses.update(self.loss_orth)

        return losses


@HEADS.register_module()
class DualMarginArcFaceMIDisBBoxHead(DualMarginArcFaceDisBBoxHead):
    """Dual-margin ArcFace head + Mutual Information (MI) regularization."""

    def __init__(self,
                 base_cpt=None,
                 base_alpha=0.5,
                 loss_mi_weight=0.001,
                 temperature=0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert base_cpt is not None, "base_cpt (pretrained base checkpoint) is required for MI alignment."
        self.base_cpt = base_cpt
        self.base_alpha = base_alpha
        self.loss_mi_weight = loss_mi_weight
        self.temperature = temperature

        # === 双分支全连接层 ===
        self.base_shared_fcs = nn.ModuleList()
        self.novel_shared_fcs = nn.ModuleList()

        for fc in self.shared_fcs:
            in_dim = fc.in_features
            out_dim = fc.out_features
            self.base_shared_fcs.append(nn.Linear(in_dim, out_dim))
            self.novel_shared_fcs.append(nn.Linear(in_dim, out_dim))
        del self.shared_fcs

        self.relu = nn.ReLU(inplace=False)
        self.loss_mi = dict()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        torch.manual_seed(0)
        processing = True
        for param_name in state_dict:
            if 'novel_shared_fcs' in param_name:
                processing = False
                break
        print('--- Processing Base Checkpoint ---', processing)

        if processing:
            base_weights = torch.load(self.base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']

            for n, p in base_weights.items():
                if 'bbox_head' not in n:
                    state_dict[n] = copy.deepcopy(p)
            for n, p in base_weights.items():
                if 'shared_fcs' in n:
                    new_n_base = n.replace('shared_fcs', 'base_shared_fcs')
                    state_dict[new_n_base] = copy.deepcopy(p)
                    new_n_novel = n.replace('shared_fcs', 'novel_shared_fcs')
                    state_dict[new_n_novel] = copy.deepcopy(p)
                if 'fc_cls' in n or 'fc_reg' in n:
                    state_dict[n] = copy.deepcopy(p)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys,
                                      unexpected_keys, error_msgs)

    def mutual_information_loss(self, base_feat, novel_feat):
        """Approximate MI using InfoNCE formulation."""
        base_feat = F.normalize(base_feat, p=2, dim=1)
        novel_feat = F.normalize(novel_feat, p=2, dim=1)
        sim_matrix = torch.matmul(base_feat, novel_feat.t()) / self.temperature

        pos = torch.diag(sim_matrix)
        loss = -torch.log(
            torch.exp(pos) / torch.exp(sim_matrix).sum(dim=1)
        ).mean()
        return loss

    def forward(self, x, labels=None):
        """Forward with dual FC branches and MI regularization."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)

        alpha = self.base_alpha
        for fc_b, fc_n in zip(self.base_shared_fcs, self.novel_shared_fcs):
            base_x = fc_b(x)
            novel_x = fc_n(x)
            x = alpha * base_x + (1 - alpha) * novel_x
            x = self.relu(x)

        # === Mutual Information regularization ===
        if self.training:
            loss_mi = self.mutual_information_loss(base_x, novel_x)
            self.loss_mi['loss_mi'] = loss_mi * self.loss_mi_weight

        # === ArcFace 分支 ===
        x_cls, x_reg = x, x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # === ArcFace 分类 ===
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)

        if labels is not None:
            dynamic_margin = self.get_dynamic_margin(labels).view(-1, 1)
            cosine_m = torch.cos(theta + dynamic_margin)
        else:
            cosine_m = torch.cos(theta + self.margin_base)

        cls_score = self.s * cosine_m
        return cls_score, bbox_pred

    def loss(self,
             cls_score, bbox_pred, rois,
             labels, label_weights,
             bbox_targets, bbox_weights,
             reduction_override=None, **kwargs):
        losses = super().loss(
            cls_score, bbox_pred, rois,
            labels, label_weights,
            bbox_targets, bbox_weights,
            reduction_override, **kwargs)
        if self.training and hasattr(self, 'loss_mi'):
            losses.update(self.loss_mi)
        return losses


@HEADS.register_module()
class DualMarginArcFaceKDMIDisBBoxHead(DualMarginArcFaceDisBBoxHead):
    """Dual-margin ArcFace head with Base/Novel dual branches + KD and MI regularization."""

    def __init__(self,
                 base_cpt=None,
                 base_alpha=0.5,
                 loss_kd_weight=0.001,
                 loss_mi_weight=0.001,
                 temperature=0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert base_cpt is not None
        self.base_cpt = base_cpt
        self.base_alpha = base_alpha
        self.loss_kd_weight = loss_kd_weight
        self.loss_mi_weight = loss_mi_weight
        self.temperature = temperature

        # === 增加双分支全连接层 ===
        self.base_shared_fcs = nn.ModuleList()
        self.novel_shared_fcs = nn.ModuleList()

        for fc in self.shared_fcs:
            in_dim = fc.in_features
            out_dim = fc.out_features
            self.base_shared_fcs.append(nn.Linear(in_dim, out_dim))
            self.novel_shared_fcs.append(nn.Linear(in_dim, out_dim))

        del self.shared_fcs

        self.loss_kd = dict()
        self.loss_mi = dict()
        self.relu = nn.ReLU(inplace=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        torch.manual_seed(0)
        processing = True
        # During training, processing the checkpoints
        # During testing, directly load the checkpoints
        for param_name in state_dict:
            if 'novel_shared_fcs' in param_name:
                processing = False
                break
        print('-------processing-----', processing)

        if processing:
            # load base training checkpoint
            num_base_classes = self.num_classes // 4 * 3
            base_weights = torch.load(self.base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']

            for n, p in base_weights.items():
                if 'bbox_head' not in n:
                    state_dict[n] = copy.deepcopy(p)

            # initialize the base branch with the weight of base training
            for n, p in base_weights.items():
                if 'shared_fcs' in n:
                    new_n_base = n.replace('shared_fcs', 'base_shared_fcs')
                    state_dict[new_n_base] = copy.deepcopy(p)
                    new_n_novel = n.replace('shared_fcs', 'novel_shared_fcs')
                    state_dict[new_n_novel] = copy.deepcopy(p)

                if 'fc_cls' in n:
                    state_dict[n] = copy.deepcopy(p)

                if not self.reg_class_agnostic and 'fc_reg' in n:
                    state_dict[n] = copy.deepcopy(p)

            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def mutual_information_loss(self, base_feat, novel_feat):
        """Approximate MI using InfoNCE formulation."""
        base_feat = F.normalize(base_feat, p=2, dim=1)
        novel_feat = F.normalize(novel_feat, p=2, dim=1)
        sim_matrix = torch.matmul(base_feat, novel_feat.t()) / self.temperature

        pos = torch.diag(sim_matrix)
        loss = -torch.log(
            torch.exp(pos) / torch.exp(sim_matrix).sum(dim=1)
        ).mean()
        return loss

    def forward(self, x, labels=None):
        """Forward with dual FC branches for base/novel learning."""
        # --- 特征提取 ---
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)

        loss_feature_kd = 0
        alpha = self.base_alpha
        base_x = x
        kd_loss_list = []
        mi_loss_list = []
        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1 - alpha) * novel_x
            kd_loss_list.append(torch.frobenius_norm(base_x - x, dim=-1))
            mi_loss_list.append(self.mutual_information_loss(base_x, novel_x))
            x = self.relu(x)
        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)

        mi_loss = torch.stack(mi_loss_list)
        mi_loss = torch.mean(mi_loss)

        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] = kd_loss

            mi_loss = mi_loss * self.loss_mi_weight
            self.loss_mi['loss_mi'] = mi_loss

        # --- 分类与回归分支 ---
        x_cls, x_reg = x, x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # === ArcFace 分类逻辑保持不变 ===
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)

        if labels is not None:
            dynamic_margin = self.get_dynamic_margin(labels).view(-1, 1)
            cosine_m = torch.cos(theta + dynamic_margin)
        else:
            cosine_m = torch.cos(theta + self.margin_base)

        cls_score = self.s * cosine_m
        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        """Compute classification, bbox, KD and Dis losses."""
        losses = super().loss(
            cls_score, bbox_pred, rois,
            labels, label_weights,
            bbox_targets, bbox_weights,
            reduction_override, **kwargs)

        # === 增加 KD loss ===
        if self.training and hasattr(self, 'loss_kd'):
            losses.update(self.loss_kd)

        # === 增加 KD loss ===
        if self.training and hasattr(self, 'loss_mi'):
            losses.update(self.loss_mi)

        return losses


@HEADS.register_module()
class PromptArcFaceDisBBoxHead(DualMarginArcFaceDisBBoxHead):
    """Dual-margin ArcFace head with Base/Novel dual branches + KD and MI regularization."""

    def __init__(self,
                 base_cpt=None,
                 base_alpha=0.5,
                 loss_kd_weight=0.001,
                 loss_mi_weight=0.001,
                 temperature=0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert base_cpt is not None
        self.base_cpt = base_cpt
        self.base_alpha = base_alpha
        self.loss_kd_weight = loss_kd_weight
        self.loss_mi_weight = loss_mi_weight
        self.temperature = temperature

        # === 增加双分支全连接层 ===
        self.base_shared_fcs = nn.ModuleList()
        self.novel_shared_fcs = nn.ModuleList()

        for fc in self.shared_fcs:
            in_dim = fc.in_features
            out_dim = fc.out_features
            self.base_shared_fcs.append(nn.Linear(in_dim, out_dim))
            self.novel_shared_fcs.append(nn.Linear(in_dim, out_dim))

        del self.shared_fcs

        self.loss_kd = dict()
        self.loss_mi = dict()
        self.relu = nn.ReLU(inplace=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        torch.manual_seed(0)
        processing = True
        # During training, processing the checkpoints
        # During testing, directly load the checkpoints
        for param_name in state_dict:
            if 'novel_shared_fcs' in param_name:
                processing = False
                break
        print('-------processing-----', processing)

        if processing:
            # load base training checkpoint
            num_base_classes = self.num_classes // 4 * 3
            base_weights = torch.load(self.base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']

            for n, p in base_weights.items():
                if 'bbox_head' not in n:
                    state_dict[n] = copy.deepcopy(p)

            # initialize the base branch with the weight of base training
            for n, p in base_weights.items():
                if 'shared_fcs' in n:
                    new_n_base = n.replace('shared_fcs', 'base_shared_fcs')
                    state_dict[new_n_base] = copy.deepcopy(p)
                    new_n_novel = n.replace('shared_fcs', 'novel_shared_fcs')
                    state_dict[new_n_novel] = copy.deepcopy(p)

                if 'fc_cls' in n:
                    state_dict[n] = copy.deepcopy(p)

                if not self.reg_class_agnostic and 'fc_reg' in n:
                    state_dict[n] = copy.deepcopy(p)

            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def mutual_information_loss(self, base_feat, novel_feat):
        """Approximate MI using InfoNCE formulation."""
        base_feat = F.normalize(base_feat, p=2, dim=1)
        novel_feat = F.normalize(novel_feat, p=2, dim=1)
        sim_matrix = torch.matmul(base_feat, novel_feat.t()) / self.temperature

        pos = torch.diag(sim_matrix)
        loss = -torch.log(
            torch.exp(pos) / torch.exp(sim_matrix).sum(dim=1)
        ).mean()
        return loss

    def forward(self, x, labels=None):
        """Forward with dual FC branches for base/novel learning."""
        # --- 特征提取 ---
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)

        loss_feature_kd = 0
        alpha = self.base_alpha
        base_x = x
        kd_loss_list = []
        mi_loss_list = []
        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1 - alpha) * novel_x
            kd_loss_list.append(1 - F.cosine_similarity(base_x, novel_x, dim=1))
            mi_loss_list.append(self.mutual_information_loss(base_x, novel_x))
            x = self.relu(x)
        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)

        mi_loss = torch.stack(mi_loss_list)
        mi_loss = torch.mean(mi_loss)

        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] = kd_loss

            mi_loss = mi_loss * self.loss_mi_weight
            self.loss_mi['loss_mi'] = mi_loss

        # --- 分类与回归分支 ---
        x_cls, x_reg = x, x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # === ArcFace 分类逻辑保持不变 ===
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)

        if labels is not None:
            dynamic_margin = self.get_dynamic_margin(labels).view(-1, 1)
            cosine_m = torch.cos(theta + dynamic_margin)
        else:
            cosine_m = torch.cos(theta + self.margin_base)

        cls_score = self.s * cosine_m
        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        """Compute classification, bbox, KD and Dis losses."""
        losses = super().loss(
            cls_score, bbox_pred, rois,
            labels, label_weights,
            bbox_targets, bbox_weights,
            reduction_override, **kwargs)

        # === 增加 KD loss ===
        if self.training and hasattr(self, 'loss_kd'):
            losses.update(self.loss_kd)

        # === 增加 KD loss ===
        if self.training and hasattr(self, 'loss_mi'):
            losses.update(self.loss_mi)

        return losses


@HEADS.register_module()
class DualMarginArcFaceKDMIRDDisBBoxHead(DualMarginArcFaceDisBBoxHead):
    """Dual-margin ArcFace head with Base/Novel dual branches + optional KD / MI / RD regularization."""

    def __init__(self,
                 base_cpt=None,
                 base_alpha=0.5,
                 # 控制 KD/MI/RD，既可通过权重也可通过开关
                 use_kd=True,
                 loss_kd_weight=0.001,
                 use_mi=True,
                 loss_mi_weight=0.001,
                 use_rd=False,
                 loss_rd_weight=0.001,
                 rd_mode='sim',  # 'sim' 表示相似矩阵 MSE；以后可扩展为 'rkd' 等
                 temperature=0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert base_cpt is not None
        self.base_cpt = base_cpt
        self.base_alpha = base_alpha

        # enable flags + weights
        self.use_kd = use_kd
        self.loss_kd_weight = loss_kd_weight

        self.use_mi = use_mi
        self.loss_mi_weight = loss_mi_weight

        self.use_rd = use_rd
        self.loss_rd_weight = loss_rd_weight
        self.rd_mode = rd_mode

        self.temperature = temperature

        # === 双分支全连接层 ===
        self.base_shared_fcs = nn.ModuleList()
        self.novel_shared_fcs = nn.ModuleList()

        for fc in self.shared_fcs:
            in_dim = fc.in_features
            out_dim = fc.out_features
            self.base_shared_fcs.append(nn.Linear(in_dim, out_dim))
            self.novel_shared_fcs.append(nn.Linear(in_dim, out_dim))

        del self.shared_fcs

        # 保留多个损失字典，训练时会注入
        self.loss_kd = dict()
        self.loss_mi = dict()
        self.loss_rd = dict()
        self.relu = nn.ReLU(inplace=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        torch.manual_seed(0)
        processing = True
        for param_name in state_dict:
            if 'novel_shared_fcs' in param_name:
                processing = False
                break
        print('-------processing-----', processing)

        if processing:
            # load base training checkpoint
            num_base_classes = self.num_classes // 4 * 3
            base_weights = torch.load(self.base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']

            for n, p in base_weights.items():
                if 'bbox_head' not in n:
                    state_dict[n] = copy.deepcopy(p)

            # initialize the base & novel branches with base training weights
            for n, p in base_weights.items():
                if 'shared_fcs' in n:
                    new_n_base = n.replace('shared_fcs', 'base_shared_fcs')
                    state_dict[new_n_base] = copy.deepcopy(p)
                    new_n_novel = n.replace('shared_fcs', 'novel_shared_fcs')
                    state_dict[new_n_novel] = copy.deepcopy(p)

                if 'fc_cls' in n:
                    state_dict[n] = copy.deepcopy(p)

                if not self.reg_class_agnostic and 'fc_reg' in n:
                    state_dict[n] = copy.deepcopy(p)

            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def mutual_information_loss(self, base_feat, novel_feat):
        """Approximate MI using InfoNCE formulation."""
        base_feat = F.normalize(base_feat, p=2, dim=1)
        novel_feat = F.normalize(novel_feat, p=2, dim=1)
        sim_matrix = torch.matmul(base_feat, novel_feat.t()) / self.temperature

        pos = torch.diag(sim_matrix)
        loss = -torch.log(
            torch.exp(pos) / torch.exp(sim_matrix).sum(dim=1)
        ).mean()
        return loss

    def intra_novel_separation_loss_labeled(self, base_feat, novel_feat, scale=5.0, margin=0.4):
        base_norm = F.normalize(base_feat, p=2, dim=1)
        novel_norm = F.normalize(novel_feat, p=2, dim=1)
        sim = torch.matmul(novel_norm, base_norm.t()) * scale
        loss = F.relu(sim - margin).mean()
        return loss

    def relation_distillation_loss(self, base_feat, novel_feat, temperature=0.1, mode='cross', scale=0.1):
        """
        Cross-Branch Relation Distillation (CB-RD, improved):
        Encourage novel features to maintain relational structure w.r.t. base space,
        while remaining stable under small-batch scenarios and avoiding negative losses.
        """

        # ---- Normalize ----
        base_norm = F.normalize(base_feat, p=2, dim=1)
        novel_norm = F.normalize(novel_feat, p=2, dim=1)

        if mode == 'cross':
            # ---- Normalize ----
            Nb, Cb = base_norm.size()
            Nn, Cn = novel_norm.size()

            if Nb == 0 or Nn == 0:
                return base_feat.new_tensor(0.0)

            # ---- Step 1: 相似度矩阵 ----
            sim_bb = torch.matmul(base_norm, base_norm.t()) / temperature  # [Nb, Nb]
            sim_bn = torch.matmul(base_norm, novel_norm.t()) / temperature  # [Nb, Nn]

            # ---- Step 2: 特征统计匹配 ----
            # 我们不逐元素对齐，而匹配统计特征（均值+方差）
            mean_bb = sim_bb.mean(dim=1, keepdim=True).detach()
            std_bb = sim_bb.std(dim=1, keepdim=True).detach()

            mean_bn = sim_bn.mean(dim=1, keepdim=True)
            std_bn = sim_bn.std(dim=1, keepdim=True)

            loss_mean = F.mse_loss(mean_bn, mean_bb)
            loss_std = F.mse_loss(std_bn, std_bb)

            # ---- Step 3: 局部结构保持（可选）----
            # 通过 top-k 最近邻相似度对齐局部结构方向（防止几何塌陷）
            k = min(3, Nb, Nn)  # 防止batch太小
            topk_idx_bb = sim_bb.topk(k=k, dim=-1).indices
            topk_idx_bn = sim_bn.topk(k=k, dim=-1).indices

            # 构造局部均值特征（不需要严格对齐）
            local_bb = base_norm[topk_idx_bb.view(-1, 1) % Nb].mean(dim=0, keepdim=True)
            local_bn = novel_norm[topk_idx_bn.view(-1, 1) % Nn].mean(dim=0, keepdim=True)
            loss_local = (1 - F.cosine_similarity(local_bb, local_bn)).mean()

            # ---- Step 4: 总损失组合 ----
            loss = scale * (loss_mean + loss_std + 0.5 * loss_local)
            loss = torch.clamp(loss, min=0.0, max=5.0)

            return loss

        elif mode == 'rkd':
            # 结构保持 (Relation Knowledge Distillation)
            d_base = torch.cdist(base_feat.detach(), base_feat.detach())
            d_novel = torch.cdist(base_feat, novel_feat)
            d_base = d_base / (d_base.mean() + 1e-6)
            d_novel = d_novel / (d_novel.mean() + 1e-6)
            loss = F.smooth_l1_loss(d_base, d_novel)
            return loss * scale

        else:
            # fallback to same-branch relational matching
            sim_base = torch.matmul(base_norm, base_norm.t())
            sim_novel = torch.matmul(novel_norm, novel_norm.t())
            mask = 1 - torch.eye(sim_base.size(0), device=sim_base.device)
            loss = F.mse_loss((sim_base * mask), (sim_novel * mask))
            return loss * scale

    def forward(self, x, labels=None):
        """Forward with dual FC branches for base/novel learning."""
        # --- 特征提取 ---
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)

        alpha = self.base_alpha
        base_x = x
        kd_loss_list = []
        mi_loss_list = []
        rd_loss_list = []

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)  # base branch feature
            novel_x = self.novel_shared_fcs[fc_ind](x)  # novel branch feature
            # 混合特征作为后续网络的输入
            x = alpha * base_x + (1.0 - alpha) * novel_x

            # kd: feature-level L2/Frobenius
            if self.use_kd:
                kd_loss_list.append(torch.norm(base_x - x, p='fro', dim=-1))

            # mi: mutual information approx (InfoNCE)
            if self.use_mi:
                mi_loss_list.append(self.intra_novel_separation_loss_labeled(base_x, novel_x))

            # rd: relation distillation (相似矩阵一致性)
            if self.use_rd:
                rd_loss_list.append(self.relation_distillation_loss(base_x, novel_x, mode=self.rd_mode))

            x = self.relu(novel_x)

        # combine per-layer losses
        kd_loss = torch.tensor(0.0, device=x.device)
        mi_loss = torch.tensor(0.0, device=x.device)
        rd_loss = torch.tensor(0.0, device=x.device)

        if len(kd_loss_list) > 0:
            kd_loss = torch.cat(kd_loss_list, dim=0).mean()
        if len(mi_loss_list) > 0:
            mi_loss = torch.stack(mi_loss_list).mean()
        if len(rd_loss_list) > 0:
            rd_loss = torch.stack(rd_loss_list).mean()

        if self.training:
            if self.use_kd:
                self.loss_kd['loss_kd'] = kd_loss * self.loss_kd_weight
            else:
                self.loss_kd.clear()

            if self.use_mi:
                self.loss_mi['loss_mi'] = mi_loss * self.loss_mi_weight
            else:
                self.loss_mi.clear()

            if self.use_rd:
                self.loss_rd['loss_rd'] = rd_loss * self.loss_rd_weight
            else:
                self.loss_rd.clear()

        # --- 分类与回归分支 ---
        x_cls, x_reg = x, x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # === ArcFace 分类逻辑保持不变 ===
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)

        if labels is not None:
            dynamic_margin = self.get_dynamic_margin(labels).view(-1, 1)
            cosine_m = torch.cos(theta + dynamic_margin)
        else:
            cosine_m = torch.cos(theta + self.margin_base)

        cls_score = self.s * cosine_m
        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        """Compute classification, bbox, KD/MI/RD regularization losses."""
        losses = super().loss(
            cls_score, bbox_pred, rois,
            labels, label_weights,
            bbox_targets, bbox_weights,
            reduction_override, **kwargs)

        # === 增加各类正则化损失（仅在 training 且对应开关打开） ===
        if self.training:
            if self.use_kd and hasattr(self, 'loss_kd'):
                losses.update(self.loss_kd)
            if self.use_mi and hasattr(self, 'loss_mi'):
                losses.update(self.loss_mi)
            if self.use_rd and hasattr(self, 'loss_rd'):
                losses.update(self.loss_rd)

        return losses


@HEADS.register_module()
class LearnableAlphaDualMarginArcFaceKDMIDisBBoxHead(DualMarginArcFaceKDMIDisBBoxHead):
    def __init__(self,
                 base_cpt=None,
                 base_alpha=0.5,
                 loss_kd_weight=0.001,
                 loss_mi_weight=0.001,
                 temperature=0.2,
                 gamma=2.0,
                 alpha_lr=0.1,
                 *args, **kwargs):
        """
        gamma: 控制alpha调节强度（类似于focal loss中的γ）
        alpha_lr: 控制alpha更新速率（平滑参数）
        """
        super().__init__(base_cpt=base_cpt,
                         base_alpha=base_alpha,
                         loss_kd_weight=loss_kd_weight,
                         loss_mi_weight=loss_mi_weight,
                         temperature=temperature,
                         *args, **kwargs)

        self.alpha_param = nn.Parameter(torch.tensor(base_alpha, dtype=torch.float))
        self.gamma = gamma
        self.alpha_lr = alpha_lr
        print(f"[Learnable+Focal Alpha Initialized] alpha={base_alpha}, gamma={gamma}")

    def forward(self, x, labels=None):
        """Forward with learnable+focal-style adaptive alpha."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)

        # === 初始 alpha ===
        alpha = torch.sigmoid(self.alpha_param)
        kd_loss_list, mi_loss_list = [], []

        base_x = x
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)

            # --- 基础融合 ---
            x = alpha * base_x + (1 - alpha) * novel_x

            # --- 计算损失 ---
            kd_val = torch.norm(base_x - x, p='fro', dim=-1).mean()
            mi_val = self.mutual_information_loss(base_x, novel_x)

            kd_loss_list.append(kd_val)
            mi_loss_list.append(mi_val)

            # --- Focal风格alpha调节 ---
            kd_sig = torch.sigmoid(kd_val.detach())
            mi_sig = torch.sigmoid(mi_val.detach())

            # “难样本”聚焦逻辑：kd大 or mi大 -> 更新alpha
            # alpha_t越接近base_x -> 越偏向稳定特征
            alpha_t = kd_sig / (kd_sig + mi_sig + 1e-6)
            alpha_t = alpha_t ** self.gamma

            # EMA平滑更新alpha
            alpha = (1 - self.alpha_lr) * alpha + self.alpha_lr * alpha_t
            alpha = torch.clamp(alpha, 0.05, 0.95)

            x = alpha * base_x + (1 - alpha) * novel_x
            x = self.relu(x)

        # === 聚合KD和MI损失 ===
        kd_loss = torch.mean(torch.stack(kd_loss_list))
        mi_loss = torch.mean(torch.stack(mi_loss_list))

        if self.training:
            self.loss_kd['loss_kd'] = kd_loss * self.loss_kd_weight
            self.loss_mi['loss_mi'] = mi_loss * self.loss_mi_weight

        # === 分类与回归分支 ===
        x_cls, x_reg = x, x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # === ArcFace 分类逻辑 ===
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)

        if labels is not None:
            dynamic_margin = self.get_dynamic_margin(labels).view(-1, 1)
            cosine_m = torch.cos(theta + dynamic_margin)
        else:
            cosine_m = torch.cos(theta + self.margin_base)

        cls_score = self.s * cosine_m
        return cls_score, bbox_pred


@HEADS.register_module()
class LearnableArcFaceBBoxHead(CosineSimBBoxHead):
    """BBox head with learnable ArcFace-style classification loss.

    This head keeps the cosine normalization of CosineSimBBoxHead,
    but replaces the classification loss with ArcFaceLoss.
    """

    def __init__(self,
                 m_init: float = 0.5,
                 s: float = 64,
                 learnable_margin: bool = True,
                 arcface_loss_cfg=dict(
                     type='ArcFaceLoss',
                     s=64.0,
                     margin=0.5,
                     loss_weight=1.0,
                     reduction='mean'),
                 *args, **kwargs):
        # super().__init__(*args, **kwargs)
        #
        # num_classes = kwargs.get("num_classes")
        #
        # # ===== 初始化 margin =====
        # if learnable_margin:
        #     # self.margin = nn.Parameter(torch.ones(num_classes + 1) * m_init)
        #     # self.margin = nn.Parameter(torch.cat([
        #     #     torch.ones(num_classes) * m_init,
        #     #     torch.zeros(1)  # background margin = 0
        #     # ]))
        #     self.margin = nn.Parameter(torch.cat([torch.tanh(torch.ones(num_classes) * m_init), torch.zeros(1)]))
        # else:
        #     self.register_buffer('margin', torch.tensor(m_init))
        #
        # self.s = s
        #
        # # ===== 初始化 ArcFaceLoss =====
        # # 可以通过 build_loss 支持 MMDet 配置风格
        # self.loss_cls = build_loss(arcface_loss_cfg)

        super().__init__(*args, **kwargs)
        # 用 self.num_classes 和 fc_cls 的输出尺寸对齐（mmdet 习惯：fc_cls.out_features = num_classes + 1）
        # 注意：在 __init__ 中 self.fc_cls 可能尚未被创建，确保在父类 __init__ 后执行这段
        C = self.fc_cls.out_features  # 包含 background
        self.bg_id = C - 1

        # 使用 raw 参数 + softplus 确保正值、稳定
        if learnable_margin:
            # 初始化 per-class raw param so margin = softplus(raw)
            init_val = float(m_init)
            raw_init = torch.log(torch.exp(torch.ones(C) * init_val) - 1.0)  # inverse softplus
            self.margin_raw = nn.Parameter(raw_init)  # shape (C,)
        else:
            # 非学可，把 margin 存 buffer（包含 bg）
            m = torch.ones(C) * float(m_init)
            m[self.bg_id] = 0.0
            self.register_buffer('margin_buffer', m)

        self.s = s
        self.loss_cls = build_loss(arcface_loss_cfg)

    # helper to get masked/clamped margin in forward/loss
    def get_margin(self):
        if hasattr(self, 'margin_raw'):
            margin = F.softplus(self.margin_raw)  # positive
            # 强制背景 margin 为 0（并且不学习）
            margin = margin.clone()
            margin[self.bg_id] = 0.0
            # optional clamp upper bound to avoid过大
            max_m = 1.2
            margin = torch.clamp(margin, 0.0, max_m)
            return margin
        else:
            return self.margin_buffer  # buffer already has bg=0

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)

        # # ====== ArcFace 分类部分 ======
        # # 特征与权重归一化
        # x_norm = F.normalize(x_cls, p=2, dim=1)
        # w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        #
        # # 基本 cosine 相似度
        # cosine = F.linear(x_norm, w_norm)
        # cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        #
        # # ✅ 在角度空间引入 learnable margin，保持梯度图完整
        # theta = torch.acos(cosine)
        # cosine_m = torch.cos(theta + self.margin)
        # cls_score = self.s * cosine_m  # 缩放输出
        #
        # return cls_score, bbox_pred

        # 特征与权重归一化
        # x_norm = F.normalize(x_cls, p=2, dim=1)
        # w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        #
        # cosine = F.linear(x_norm, w_norm)  # shape (N, C)
        # cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        # cls_score = self.s * cosine  # **不加 margin**
        # return cls_score, bbox_pred
        # 归一化特征和权重（不要 in-place 修改 fc_cls.weight）
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)  # 保留梯度路径
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)  # (N, C)
        cls_score = self.s * cosine  # 先不加 margin，这里只是 base logits
        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        """Compute classification and bbox losses using ArcFaceLoss."""

        losses = dict()

        # ===== 分类分支 =====
        if cls_score is not None and cls_score.numel() > 0:
            # avg_factor 可以根据 label_weights 自动计算
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)

            # # 调用 ArcFaceLoss
            # losses['loss_cls'] = self.loss_cls(
            #     cls_score,
            #     labels,
            #     weight=label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override,
            #     margin=self.margin  # 传入可学习 margin
            # )
            #
            # # 计算分类准确率（可选）
            # losses['acc'] = accuracy(cls_score, labels)
            ########################################################################
            # cosine = cls_score / self.s
            # cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
            # theta = torch.acos(cosine)
            #
            # # === 类自适应 margin，仅对目标类作用 ===
            # cosine_m = cosine.clone()
            # margin_per_sample = self.margin[labels]
            # theta_y = theta[torch.arange(len(labels)), labels]
            # theta_y_m = theta_y + margin_per_sample
            # cosine_y_m = torch.cos(theta_y_m)
            # cosine_m[torch.arange(len(labels)), labels] = cosine_y_m
            #
            # cls_score_m = self.s * cosine_m
            #
            # losses['loss_cls'] = self.loss_cls(
            #     cls_score_m,
            #     labels,
            #     weight=label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override)
            #
            # losses['acc'] = accuracy(cls_score, labels)
            #######################################################################
            # 基本 cosine（恢复到角度域之前）
            # cosine = (cls_score / self.s).clamp(-1 + 1e-7, 1 - 1e-7)  # (N, C)
            # theta = torch.acos(cosine)  # (N, C)
            #
            # # 只对正样本且为前景类的样本应用 margin
            # bg_class_ind = self.num_classes
            # pos_mask = (labels >= 0) & (labels < bg_class_ind)  # boolean mask N
            # cosine_m = cosine.clone()
            #
            # if pos_mask.any():
            #     idx = pos_mask.nonzero(as_tuple=False).squeeze(1)  # indices of positive samples
            #     labels_pos = labels[idx]  # 对应的 class id
            #     # 使用被约束/裁剪的 margin：如 clamp 到 [0, max_m]
            #     max_m = 0.5
            #     margin_all = torch.clamp(self.margin, 0.0, max_m).to(cosine.device)  # (C,)
            #     margin_per_sample = margin_all[labels_pos]  # (num_pos,)
            #
            #     theta_y = theta[idx, labels_pos]  # (num_pos,)
            #     theta_y_m = theta_y + margin_per_sample
            #     cosine_y_m = torch.cos(theta_y_m)
            #
            #     cosine_m[idx, labels_pos] = cosine_y_m
            #
            # cls_score_m = self.s * cosine_m
            #
            # losses['loss_cls'] = self.loss_cls(
            #     cls_score_m,
            #     labels,
            #     weight=label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override)
            #
            # # 若要监控原始 acc，仍用未加 margin 的 cls_score
            # losses['acc'] = accuracy(cls_score, labels)

            # 将 logits 变回 cosine 域
            # cosine = (cls_score / self.s).clamp(-1 + 1e-7, 1 - 1e-7)  # (N, C)
            # theta = torch.acos(cosine)  # (N, C)
            #
            # # 取出 margin（Tensor length C）
            # margin_all = self.get_margin().to(cosine.device)  # (C,), bg 已经为 0
            #
            # # 只对正样本且为前景的样本应用 margin
            # bg_class_ind = self.bg_id
            # pos_mask = (labels >= 0) & (labels < bg_class_ind)
            # cosine_m = cosine.clone()
            #
            # if pos_mask.any():
            #     pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)
            #     labels_pos = labels[pos_idx]  # 这些标签必然 < bg_class_ind
            #     # safe indexing: labels_pos 应在 [0, C-2]
            #     # 取对应类的 margin 值
            #     m_per_sample = margin_all[labels_pos]  # (num_pos,)
            #     theta_y = theta[pos_idx, labels_pos]
            #     theta_y_m = theta_y + m_per_sample
            #     cosine_y_m = torch.cos(theta_y_m)
            #     cosine_m[pos_idx, labels_pos] = cosine_y_m
            #
            # cls_score_m = self.s * cosine_m
            #
            # losses['loss_cls'] = self.loss_cls(
            #     cls_score_m,
            #     labels,
            #     weight=label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override
            # )
            # losses['acc'] = accuracy(cls_score, labels)

            cosine = (cls_score / self.s).clamp(-1 + 1e-7, 1 - 1e-7)  # (N, C)

            # margin per class (Tensor shape (C,))
            margin_all = self.get_margin().to(cosine.device)  # bg already 0

            # make a copy; we will only modify target logits
            cosine_m = cosine.clone()

            # background index (the last index)
            bg_id = self.bg_id

            # positive mask: foreground samples (exclude bg/ignore)
            pos_mask = (labels >= 0) & (labels < bg_id)
            if pos_mask.any():
                pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)  # (P,)
                labels_pos = labels[pos_idx]  # (P,)

                # gather the target cosine values for positive samples
                cosine_y = cosine[pos_idx, labels_pos]  # (P,)

                # numerical stability for acos
                cosine_y = cosine_y.clamp(-1 + 1e-7, 1 - 1e-7)

                # compute theta only for the true class entries
                theta_y = torch.acos(cosine_y)  # (P,)

                # per-sample margin
                m_per_sample = margin_all[labels_pos]  # (P,)

                # apply margin in angle domain
                theta_y_m = theta_y + m_per_sample
                cosine_y_m = torch.cos(theta_y_m)

                # assign back into the copy
                cosine_m[pos_idx, labels_pos] = cosine_y_m

            # scale back to logits
            cls_score_m = self.s * cosine_m

            # compute classification loss (ArcFace-style applied)
            losses['loss_cls'] = self.loss_cls(
                cls_score_m,
                labels,
                weight=label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override
            )
            losses['acc'] = accuracy(cls_score, labels)

            ########################################################################

        # ===== 回归分支 =====
        # if bbox_pred is not None:
        #     loss_bbox = self.loss_bbox(
        #         bbox_pred,
        #         bbox_targets,
        #         bbox_weights,
        #         reduction_override=reduction_override)
        #     losses['loss_bbox'] = loss_bbox
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses


@HEADS.register_module()
class LearnableGroupArcFaceBBoxHead(CosineSimBBoxHead):
    """ArcFace BBox Head with learnable group-wise margins for base and novel classes."""

    def __init__(self,
                 m_init: float = 0.5,
                 s: float = 64,
                 arcface_loss_cfg=dict(
                     type='ArcFaceLoss',
                     s=64.0,
                     margin=0.5,
                     loss_weight=1.0,
                     reduction='mean'),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 类别分组信息
        C = self.fc_cls.out_features  # num_classes + 1 (含背景)
        self.bg_id = C - 1
        self.num_base_classes = (C - 1) // 4 * 3  # 3/4 base + 1/4 novel

        # 可学习 margin 参数
        self.margin_base_raw = nn.Parameter(torch.log(torch.exp(torch.tensor(m_init)) - 1.0))
        self.margin_novel_raw = nn.Parameter(torch.log(torch.exp(torch.tensor(m_init * 0.8)) - 1.0))

        # scale
        self.s = s
        self.loss_cls = build_loss(arcface_loss_cfg)

    def get_group_margins(self):
        """Softplus activation to ensure margin > 0 and bounded."""
        m_base = F.softplus(self.margin_base_raw).clamp(0.0, 1.2)
        m_novel = F.softplus(self.margin_novel_raw).clamp(0.0, 1.2)
        return m_base, m_novel

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # standard forward (与 CosineSimBBoxHead 一致)
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls, x_reg = x, x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        # cosine similarity
        x_norm = F.normalize(x_cls, p=2, dim=1)
        w_norm = F.normalize(self.fc_cls.weight, p=2, dim=1)
        cosine = F.linear(x_norm, w_norm).clamp(-1 + 1e-7, 1 - 1e-7)
        cls_score = self.s * cosine

        # ✅ Dummy usage of margins for DDP safety
        if self.training:
            m_base, m_novel = self.get_group_margins()
            cls_score = cls_score + 0.0 * (m_base + m_novel)  # 不影响前向结果，但参与计算图

        return cls_score, bbox_pred

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             **kwargs):
        losses = dict()
        if cls_score is not None and cls_score.numel() > 0:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            cosine = (cls_score / self.s).clamp(-1 + 1e-7, 1 - 1e-7)
            cosine_m = cosine.clone()
            m_base, m_novel = self.get_group_margins()

            bg_id = self.bg_id
            pos_mask = (labels >= 0) & (labels < bg_id)
            if pos_mask.any():
                pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1)
                labels_pos = labels[pos_idx]

                num_base_classes = self.num_base_classes
                base_inds = labels_pos < num_base_classes
                novel_inds = (labels_pos >= num_base_classes) & (labels_pos < bg_id)

                cosine_y = cosine[pos_idx, labels_pos]
                theta_y = torch.acos(cosine_y.clamp(-1 + 1e-7, 1 - 1e-7))
                theta_y_m = theta_y.clone()

                if base_inds.any():
                    theta_y_m[base_inds] = theta_y[base_inds] + m_base
                if novel_inds.any():
                    theta_y_m[novel_inds] = theta_y[novel_inds] + m_novel

                cosine_y_m = torch.cos(theta_y_m)
                cosine_m[pos_idx, labels_pos] = cosine_y_m

            cls_score_m = self.s * cosine_m

            losses['loss_cls'] = self.loss_cls(
                cls_score_m, labels, weight=label_weights,
                avg_factor=avg_factor, reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)

        # bbox loss identical to parent
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1, 4)[pos_inds, labels[pos_inds]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred, bbox_targets[pos_inds],
                    bbox_weights[pos_inds], avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        return losses
