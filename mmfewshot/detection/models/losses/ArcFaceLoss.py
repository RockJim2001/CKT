# ArcFaceLoss 实现，引自 https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf
# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss, LOSSES


#
#
#
#
# # def arcface_loss(pred,
# #                  label,
# #                  s=64.0,
# #                  margin=0.5,
# #                  weight=None,
# #                  reduction='mean',
# #                  avg_factor=None,
# #                  class_weight=None,
# #                  ignore_index=-100,
# #                  easy_margin=False):
# #     """ArcFace Loss.
# #
# #     Args:
# #         pred (torch.Tensor): Cosine logits with shape (N, C), each element is cos(θ).
# #         label (torch.Tensor): Ground truth labels with shape (N,).
# #         s (float): Norm scaling factor.
# #         margin (float): Additive angular margin.
# #         weight (torch.Tensor, optional): Sample-wise loss weight.
# #         reduction (str, optional): Loss reduction method.
# #         avg_factor (int, optional): Averaging factor for the loss.
# #         class_weight (list[float], optional): Class-wise loss weight.
# #         ignore_index (int | None): Index to be ignored in labels.
# #         easy_margin (bool): Whether to use easy margin.
# #
# #     Returns:
# #         torch.Tensor: Calculated loss value.
# #     """
# #
# #     # 对权重和特征进行L2归一化
# #     cos_m = math.cos(margin)
# #     sin_m = math.sin(margin)
# #     th = math.cos(math.pi - margin)
# #     mm = math.sin(math.pi - margin) * margin
# #
# #     one_hot = torch.zeros_like(pred)
# #     valid_mask = (label != ignore_index)
# #     label_valid = label[valid_mask]
# #     pred_valid = pred[valid_mask]
# #     if label_valid.numel() == 0:
# #         return (pred.sum() * 0.0).clone()
# #
# #     index = torch.arange(0, label_valid.size(0), dtype=torch.long, device=pred.device)
# #     one_hot[valid_mask, label_valid] = 1.0
# #     cosine = pred_valid.clamp(-1 + 1e-6, 1 - 1e-6)  # 防止sqrt出现负数
# #     sine = torch.sqrt(1.0 - cosine ** 2 + 1e-6)
# #     phi = cosine * cos_m - sine * sin_m
# #
# #     if easy_margin:
# #         phi = torch.where(cosine > 0, phi, cosine)
# #     else:
# #         phi = torch.where(cosine > th, phi, cosine - mm)
# #
# #     output = pred.clone()
# #     output[valid_mask] = pred_valid * (1 - one_hot[valid_mask]) + phi * one_hot[valid_mask]
# #     output = output * s
# #
# #     loss = F.cross_entropy(
# #         output,
# #         label,
# #         weight=class_weight,
# #         reduction='none',
# #         ignore_index=ignore_index
# #     )
# #
# #     if weight is not None:
# #         weight = weight.float()
# #     loss = weight_reduce_loss(
# #         loss, weight=weight, reduction=reduction, avg_factor=avg_factor
# #     )
# #     return loss
# import torch
# import torch.nn.functional as F
# import math
# from mmdet.models.losses.utils import weight_reduce_loss
#
#
# def arcface_loss(pred,
#                  label,
#                  s=64.0,
#                  margin=0.5,
#                  weight=None,
#                  reduction='mean',
#                  avg_factor=None,
#                  class_weight=None,
#                  ignore_index=-100,
#                  easy_margin=False):
#     """
#     ArcFace Loss for MMDetection compatible interface.
#     Args:
#         pred (torch.Tensor): cosine logits (N, C)
#         label (torch.Tensor): ground truth labels (N,)
#         s (float): scale
#         margin (float): additive angular margin
#         weight (torch.Tensor, optional): sample-wise weight
#         reduction (str, optional): 'mean' or 'sum'
#         avg_factor (int, optional): averaging factor
#         class_weight (list or tensor, optional): class-wise weight
#         ignore_index (int, optional): label index to ignore
#         easy_margin (bool): whether to use easy margin
#     Returns:
#         torch.Tensor: loss value
#     """
#     # --- Step0: valid mask ---
#     valid_mask = (label != ignore_index) & (label >= 0)
#     if valid_mask.sum() == 0:
#         return (pred.sum() * 0.0).clone()
#
#     pred_valid = pred[valid_mask]
#     label_valid = label[valid_mask]
#
#     # 数值稳定
#     cosine = pred_valid / s
#     cosine = cosine.clamp(-1 + 1e-4, 1 - 1e-4)
#
#     # --- Step1: compute phi ---
#     cos_m = math.cos(margin)
#     sin_m = math.sin(margin)
#     th = math.cos(math.pi - margin)
#     mm = math.sin(math.pi - margin) * margin
#
#
#     sine = torch.sqrt(1.0 - cosine ** 2)
#     phi = cosine * cos_m - sine * sin_m
#     if easy_margin:
#         phi = torch.where(cosine > 0, phi, cosine)
#     else:
#         phi = torch.where(cosine > th, phi, cosine - mm)
#
#     # # --- Step2: one-hot safely ---
#     # one_hot = F.one_hot(label_valid, num_classes=pred.size(1)).float()
#     #
#     # # --- Step3: combine ---
#     # output = pred.clone()
#     # output[valid_mask] = cosine * (1 - one_hot) + phi * one_hot
#     # output = output * s
#     # # output = pred.clone()
#     # # output[valid_mask] = pred_valid
#     # # output[valid_mask, label_valid] = phi[torch.arange(label_valid.size(0)), label_valid]
#     # # output = output * s
#     #
#     # label = label.clone()
#     # label[label >= pred.size(1)] = ignore_index
#     #
#     # # --- Step4: compute CE loss ---
#     # loss = F.cross_entropy(
#     #     output,
#     #     label,
#     #     weight=class_weight,
#     #     reduction='none',
#     #     ignore_index=ignore_index
#     # )
#     #
#     # if weight is not None:
#     #     loss = loss * weight.float()
#     #
#     # # --- Step5: reduction ---
#     # loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
#     # return loss
#
#     # --- Step2: replace target logits ---
#     # output = cosine.clone()
#     # idx = torch.arange(label_valid.size(0), device=label.device)
#     #
#     # background_id = pred_valid.size(1) - 1
#     #
#     # for i in range(label_valid.size(0)):
#     #     if label_valid[i] != background_id:  # 仅前景加 margin
#     #         output[i, label_valid[i]] = phi[i, label_valid[i]]
#     #
#     # # --- Step3: scale ---
#     # output = output * s
#     #
#     # # --- Step4: compute CE loss ---
#     # loss = F.cross_entropy(
#     #     output,
#     #     label_valid,
#     #     weight=class_weight,
#     #     reduction='none'
#     # )
#     #
#     # if weight is not None:
#     #     loss = loss * weight[valid_mask].float()
#     #
#     # # --- Step5: reduction ---
#     # loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
#     # return loss
#
#     # --- Step3: prepare output logits (use cosine for non-targets; phi for targets) ---
#     # 注意 background_id 应该是最后一个 index (C-1) 当你是 num_classes+1 格式
#     C = pred.size(1)
#     background_id = C - 1  # 通常 mmdet 的 cls 输出把背景放在最后，如果你改了顺序请调整
#
#     # one-hot for targets (safe)
#     # 但要确保 label_valid < C，否则需要把越界 label 设为 ignore_index
#     # 将越界标签设为 ignore
#     label_tmp = label_valid.clone()
#     label_tmp[label_tmp >= C] = ignore_index
#     valid_idx_mask = label_tmp != ignore_index
#     if valid_idx_mask.sum() == 0:
#         return (pred.sum() * 0.0).clone()
#
#     # build output starting from cosine
#     output = cosine.clone()  # still in cosine domain
#     idx = torch.arange(label_valid.size(0), device=label_valid.device)
#
#     # only apply margin to true foreground (i.e. labels that are not background_id and not ignore)
#     for i in range(label_valid.size(0)):
#         lab = int(label_valid[i].item())
#         if lab == ignore_index:
#             continue
#         if lab != background_id:
#             # safe: lab in [0, C-1]
#             output[i, lab] = phi[i, lab]
#         else:
#             # background: keep cosine (no margin)
#             output[i, lab] = cosine[i, lab]
#
#     # --- Step4: scale back to logits domain (s * cosine_or_phi) ---
#     output = output * s
#
#     # --- Step5: compute CE loss using label_valid (but need to mask out invalid labels) ---
#     # Prepare labels for CE: those that were >= C were set to ignore before
#     ce_labels = label_valid.clone()
#     ce_labels[ce_labels >= C] = ignore_index
#
#     loss = F.cross_entropy(
#         output,
#         ce_labels,
#         weight=class_weight,
#         reduction='none',
#         ignore_index=ignore_index
#     )
#
#     if weight is not None:
#         # weight is per-sample weight aligned with original batch; need to index with valid_mask
#         loss = loss * weight[valid_mask].float()
#
#     loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
#     return loss
#
#
# @LOSSES.register_module()
# class ArcFaceLoss(nn.Module):
#     """ArcFace Loss Module (mmdet-compatible).
#
#     Args:
#         s (float): Norm scaling factor. Defaults to 64.0.
#         margin (float): Additive angular margin. Defaults to 0.5.
#         reduction (str): Loss reduction method. Defaults to 'mean'.
#         class_weight (list[float], optional): Weight for each class. Defaults to None.
#         ignore_index (int | None): Label index to ignore. Defaults to -100.
#         loss_weight (float): Loss weight. Defaults to 1.0.
#         easy_margin (bool): Whether to use easy margin. Defaults to False.
#     """
#
#     def __init__(self,
#                  s=64.0,
#                  margin=0.5,
#                  reduction='mean',
#                  use_sigmoid=False,
#                  use_mask=False,
#                  class_weight=None,
#                  ignore_index=-100,
#                  loss_weight=1.0,
#                  easy_margin=False):
#         super(ArcFaceLoss, self).__init__()
#         self.s = s
#         self.margin = margin
#         self.reduction = reduction
#         self.use_sigmoid = use_sigmoid
#         self.use_mask = use_mask
#         self.class_weight = class_weight
#         self.ignore_index = ignore_index
#         self.loss_weight = loss_weight
#         self.easy_margin = easy_margin
#
#     def forward(self,
#                 cls_score,
#                 label,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 ignore_index=None,
#                 **kwargs):
#         reduction = reduction_override if reduction_override else self.reduction
#         ignore_index = ignore_index if ignore_index is not None else self.ignore_index
#         class_weight = None
#         if self.class_weight is not None:
#             class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)
#         loss = arcface_loss(
#             cls_score,
#             label,
#             s=self.s,
#             margin=self.margin,
#             weight=weight,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             class_weight=class_weight,
#             ignore_index=ignore_index,
#             easy_margin=self.easy_margin
#         )
#         # print("当前的loss是：" + str(loss.item()))
#         return self.loss_weight * loss
#
#
# def arcface_loss_class_adaptive(pred,
#                                 label,
#                                 s=64.0,
#                                 base_margin=0.5,
#                                 novel_margin=0.35,
#                                 base_classes=None,
#                                 weight=None,
#                                 reduction='mean',
#                                 avg_factor=None,
#                                 class_weight=None,
#                                 ignore_index=-100,
#                                 easy_margin=False):
#     """ArcFace Loss with class-adaptive margin.
#
#     Args:
#         pred (torch.Tensor): Cosine logits with shape (N, C).
#         label (torch.Tensor): Ground truth labels with shape (N,).
#         s (float): Norm scaling factor.
#         base_margin (float): Margin for base classes.
#         novel_margin (float): Margin for novel classes.
#         base_classes (list[int]): List of base class indices.
#         weight (torch.Tensor, optional): Sample-wise loss weight.
#         reduction (str, optional): Loss reduction method.
#         avg_factor (int, optional): Averaging factor for the loss.
#         class_weight (list[float], optional): Class-wise loss weight.
#         ignore_index (int | None): Index to be ignored in labels.
#         easy_margin (bool): Whether to use easy margin.
#     """
#     one_hot = torch.zeros_like(pred)
#     valid_mask = (label != ignore_index)
#     label_valid = label[valid_mask]
#     pred_valid = pred[valid_mask]
#     if label_valid.numel() == 0:
#         return (pred.sum() * 0.0).clone()
#
#     # 构造 margin tensor (不同类别有不同 margin)
#     margin_tensor = torch.zeros_like(pred_valid)
#     if base_classes is not None:
#         base_classes = set(base_classes)
#         margins = torch.full_like(label_valid, novel_margin, dtype=torch.float, device=pred.device)
#         for i, cls in enumerate(label_valid):
#             if int(cls.item()) in base_classes:
#                 margins[i] = base_margin
#     else:
#         # 如果没有指定 base_classes，就统一用 base_margin
#         margins = torch.full_like(label_valid, base_margin, dtype=torch.float, device=pred.device)
#
#     cos_m = torch.cos(margins)
#     sin_m = torch.sin(margins)
#     th = torch.cos(math.pi - margins)
#     mm = torch.sin(math.pi - margins) * margins
#
#     index = torch.arange(0, label_valid.size(0), dtype=torch.long, device=pred.device)
#     one_hot[valid_mask, label_valid] = 1.0
#     cosine = pred_valid.clamp(-1 + 1e-6, 1 - 1e-6)
#     sine = torch.sqrt(1.0 - cosine ** 2 + 1e-6)
#     phi = cosine * cos_m - sine * sin_m
#
#     if easy_margin:
#         phi = torch.where(cosine > 0, phi, cosine)
#     else:
#         phi = torch.where(cosine > th, phi, cosine - mm)
#
#     output = pred.clone()
#     output[valid_mask] = pred_valid * (1 - one_hot[valid_mask]) + phi * one_hot[valid_mask]
#     output = output * s
#
#     loss = F.cross_entropy(
#         output,
#         label,
#         weight=class_weight,
#         reduction='none',
#         ignore_index=ignore_index
#     )
#
#     if weight is not None:
#         weight = weight.float()
#     loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
#     return loss
#
#
# @LOSSES.register_module()
# class ArcFaceLossAdaptive(nn.Module):
#     """ArcFace Loss with base/novel adaptive margins (mmdet-compatible)."""
#
#     def __init__(self,
#                  s=64.0,
#                  base_margin=0.5,
#                  novel_margin=0.35,
#                  base_classes=None,
#                  reduction='mean',
#                  use_sigmoid=False,
#                  use_mask=False,
#                  class_weight=None,
#                  ignore_index=-100,
#                  loss_weight=1.0,
#                  easy_margin=False):
#         super(ArcFaceLossAdaptive, self).__init__()
#         self.s = s
#         self.base_margin = base_margin
#         self.novel_margin = novel_margin
#         self.base_classes = base_classes if base_classes is not None else []
#         self.reduction = reduction
#         self.use_sigmoid = use_sigmoid
#         self.use_mask = use_mask
#         self.class_weight = class_weight
#         self.ignore_index = ignore_index
#         self.loss_weight = loss_weight
#         self.easy_margin = easy_margin
#
#     def forward(self,
#                 cls_score,
#                 label,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 ignore_index=None,
#                 **kwargs):
#         reduction = reduction_override if reduction_override else self.reduction
#         ignore_index = ignore_index if ignore_index is not None else self.ignore_index
#         class_weight = None
#         if self.class_weight is not None:
#             class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)
#         loss = arcface_loss_class_adaptive(
#             cls_score,
#             label,
#             s=self.s,
#             base_margin=self.base_margin,
#             novel_margin=self.novel_margin,
#             base_classes=self.base_classes,
#             weight=weight,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             class_weight=class_weight,
#             ignore_index=ignore_index,
#             easy_margin=self.easy_margin
#         )
#         return self.loss_weight * loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmdet.models import LOSSES


@LOSSES.register_module()
class ClasswiseMarginArcFaceLoss(nn.Module):
    """ArcFace Loss with class-wise learnable margins (with background support)."""

    def __init__(self,
                 num_classes,
                 s=64.0,
                 init_margin=0.5,
                 bg_margin=0.0,  # 背景类 margin
                 learnable_margin=True,
                 easy_margin=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=-100,
                 use_sigmoid=False,
                 loss_weight=1.0):
        super().__init__()
        self.s = s
        self.easy_margin = easy_margin
        self.reduction = reduction
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.num_classes = num_classes  # foreground 类别数，不包含背景
        self.total_classes = num_classes + 1  # 包含背景类

        if learnable_margin:
            # 每个类一个 margin，包括背景类
            margins = torch.ones(self.total_classes) * init_margin
            margins[-1] = bg_margin  # 背景类 margin 初始值（可为 0）
            self.m_raw = nn.Parameter(margins)
        else:
            margins = torch.ones(self.total_classes) * init_margin
            margins[-1] = bg_margin
            self.register_buffer("margin", margins)

    def get_margin(self):
        if hasattr(self, "m_raw"):
            # 限制 margin 范围到 (0, π/2)
            return torch.sigmoid(self.m_raw) * (math.pi / 2)
        else:
            return self.margin

    def forward(self,
                pred,
                target,
                label_weights=None,
                avg_factor=None,
                reduction_override=None):
        """pred: [N, C+1], target: [N]"""

        cosine = pred.clamp(-1.0, 1.0)
        margins = self.get_margin()  # shape: [C+1]

        # 有效样本 mask
        valid_mask = (target != self.ignore_index) & (target < self.total_classes)
        if not valid_mask.any():
            return cosine.new_tensor(0.0)

        target_valid = target[valid_mask]

        # 取当前 batch 中每个样本对应类的 margin
        margin = margins[target_valid].unsqueeze(1)  # [N_valid, 1]

        cos_m = torch.cos(margin)
        sin_m = torch.sin(margin)
        th = torch.cos(math.pi - margin)
        mm = torch.sin(math.pi - margin) * margin

        cosine_valid = cosine[valid_mask]
        sine = torch.sqrt((1.0 - cosine_valid.pow(2)).clamp(0, 1))
        phi = cosine_valid * cos_m - sine * sin_m

        if self.easy_margin:
            phi = torch.where(cosine_valid > 0, phi, cosine_valid)
        else:
            phi = torch.where(cosine_valid > th, phi, cosine_valid - mm)

        # --- one-hot ---
        one_hot = torch.zeros_like(cosine_valid)
        one_hot.scatter_(1, target_valid.view(-1, 1), 1.0)

        # 背景类的 margin 可设为 0（实际即不改变原 cosine）
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_valid)
        output *= self.s

        # 将有效样本放回原 batch 尺寸
        full_output = cosine.clone()
        full_output[valid_mask] = output

        # --- cross entropy ---
        loss = F.cross_entropy(full_output,
                               target,
                               weight=self.class_weight,
                               reduction=self.reduction,
                               ignore_index=self.ignore_index)

        return self.loss_weight * loss


@LOSSES.register_module()
class LearningMarginArcFaceLoss(nn.Module):
    """ArcFace Loss with learnable margin (stable version)"""

    def __init__(self,
                 s=64.0,
                 init_margin=0.5,
                 learnable_margin=True,
                 easy_margin=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=-100,
                 use_sigmoid=False,
                 loss_weight=1.0):
        super().__init__()
        self.s = s
        self.easy_margin = easy_margin
        self.reduction = reduction
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

        if learnable_margin:
            # 使用 Sigmoid 限制 margin 范围在 (0, π/2)
            self.m_raw = nn.Parameter(torch.tensor(init_margin))
        else:
            self.register_buffer("margin", torch.tensor(init_margin))

    def get_margin(self):
        if hasattr(self, "m_raw"):
            return torch.sigmoid(self.m_raw) * (math.pi / 2)
        else:
            return self.margin

    def forward(self,
                pred,
                target,
                label_weights,
                avg_factor=None,
                reduction_override=None):
        cosine = pred.clamp(-1.0, 1.0)
        margin = self.get_margin()

        cos_m = torch.cos(margin)
        sin_m = torch.sin(margin)
        th = torch.cos(math.pi - margin)
        mm = torch.sin(math.pi - margin) * margin

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * cos_m - sine * sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > th, phi, cosine - mm)

        # one-hot
        one_hot = torch.zeros_like(pred)
        valid_mask = (target != self.ignore_index) & (target < pred.size(1))
        if valid_mask.any():
            one_hot[valid_mask] = F.one_hot(
                target[valid_mask], num_classes=pred.size(1)
            ).float()

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = F.cross_entropy(output, target,
                               weight=self.class_weight,
                               reduction=self.reduction,
                               ignore_index=self.ignore_index)

        return self.loss_weight * loss


@LOSSES.register_module()
class MarginArcFaceLoss(nn.Module):
    """ArcFace Loss (MMDetection compatible with CrossEntropyLoss interface).
    使用固定的margin
    """

    def __init__(self,
                 s=64.0,
                 margin=0.5,
                 easy_margin=False,
                 use_sigmoid=False,   # 仅为兼容接口
                 reduction='mean',
                 class_weight=None,
                 ignore_index=-100,
                 loss_weight=1.0):
        super().__init__()
        self.s = s
        self.margin = margin
        self.easy_margin = easy_margin
        self.use_sigmoid = use_sigmoid  # 只是占位，不会真正使用
        self.reduction = reduction
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self,
                pred,
                target,
                label_weights,
                avg_factor=None,
                reduction_override=None):
        """ pred: cosine logits, shape (N, C)
            target: class indices, shape (N,)
        """
        assert pred.dim() == 2, "ArcFace input must be (N, C)"
        num_classes = pred.size(1)

        # --- Step1: apply margin ---
        cosine = pred.clamp(-1.0, 1.0)  # safety
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --- Step2: one-hot encode (skip ignore_index) ---
        with torch.no_grad():
            one_hot = torch.zeros_like(pred)
            valid_mask = (target != self.ignore_index) & (target < num_classes)
            if valid_mask.any():
                one_hot[valid_mask] = F.one_hot(
                    target[valid_mask], num_classes=num_classes).float()

        # --- Step3: logits replace ---
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  # scale

        # --- Step4: CE loss ---
        loss = F.cross_entropy(
            output,
            target,
            weight=self.class_weight,
            reduction='none',
            ignore_index=self.ignore_index,
        )

        if self.reduction == 'mean':
            if avg_factor is None:
                loss = loss.mean()
            else:
                loss = loss.sum() / avg_factor
        elif self.reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss


def arcface_loss(pred,
                 label,
                 s=64.0,
                 margin=None,  # 已在 forward 阶段使用
                 weight=None,
                 reduction='mean',
                 avg_factor=None,
                 class_weight=None,
                 ignore_index=-100,
                 **kwargs):
    """Simplified ArcFace Loss (no repeated margin computation)."""
    valid_mask = (label != ignore_index) & (label >= 0)
    if valid_mask.sum() == 0:
        return (pred.sum() * 0.0).clone()

    pred_valid = pred[valid_mask]
    label_valid = label[valid_mask]

    # 标准交叉熵（pred 已经包含 margin 逻辑）
    loss = F.cross_entropy(
        pred_valid,
        label_valid,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index
    )

    if weight is not None:
        loss = loss * weight[valid_mask].float()

    # 平均或加权
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module()
class ArcFaceLoss(nn.Module):
    def __init__(self,
                 s=64.0,
                 margin=0.5,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=-100,
                 loss_weight=1.0,
                 **kwargs):
        super().__init__()
        self.s = s
        self.margin = margin
        self.reduction = reduction
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        reduction = reduction_override if reduction_override else self.reduction
        ignore_index = ignore_index if ignore_index is not None else self.ignore_index

        class_weight = None
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)

        loss = arcface_loss(
            cls_score,
            label,
            s=self.s,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            class_weight=class_weight,
            ignore_index=ignore_index
        )

        return self.loss_weight * loss
