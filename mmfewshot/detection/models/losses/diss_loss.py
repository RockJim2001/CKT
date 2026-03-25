import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class DisLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 shot,
                 loss_base_margin_weight=None,
                 loss_novel_margin_weight=None,
                 loss_neg_margin_weight=0.001,
                 max_diff=1.,
                 max_loss=1.,
                 ignore_neg=False,
                 reduction='sum',
                 loss_weight=1.0,
                 power_weight=2.0):
        super(DisLoss, self).__init__()
        self.max_diff = max_diff
        self.max_loss = max_loss
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.power_weight = power_weight

        # beta
        if loss_novel_margin_weight is None:
            self.loss_novel_margin_weight = 1. / shot
        else:
            self.loss_novel_margin_weight = loss_novel_margin_weight

        # alpha
        if loss_base_margin_weight is None:
            self.loss_base_margin_weight = self.loss_novel_margin_weight / 3.
        else:
            self.loss_base_margin_weight = loss_base_margin_weight

        # gamma
        self.loss_neg_margin_weight = loss_neg_margin_weight

        self.num_classes = num_classes
        self.ignore_neg = ignore_neg

    def forward(self, cls_score, labels, label_weights=None, decay_rate=None, **kwargs):
        losses = dict()

        loss_base_margin_weight = self.loss_base_margin_weight
        loss_novel_margin_weight = self.loss_novel_margin_weight
        loss_neg_margin_weight = self.loss_neg_margin_weight

        if decay_rate is not None:
            loss_base_margin_weight = self.loss_base_margin_weight * decay_rate
            loss_novel_margin_weight = self.loss_novel_margin_weight * decay_rate
            loss_neg_margin_weight = self.loss_neg_margin_weight * decay_rate

        num_base_classes = self.num_classes // 4 * 3
        base_inds = labels < num_base_classes
        novel_inds = (labels >= num_base_classes) & (labels < self.num_classes)
        base_labels = labels[base_inds]
        novel_labels = labels[novel_inds]
        scores = cls_score.softmax(-1)
        base_scores = scores[base_inds, base_labels]
        novel_scores = scores[novel_inds, novel_labels]  # 分数即为cos距离

        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)

        # base margin
        num_base = base_scores.size(0)
        if num_base > 0:
            loss_base_margin = []
            for label in base_labels.unique():
                l_inds = labels == label
                l_scores = scores[l_inds, label]
                l_all_scores = scores[l_inds]
                mask = torch.ones_like(l_all_scores).bool()
                mask[:, label] = False
                if self.ignore_neg:
                    mask[:, -1] = False
                l_other_scores = l_all_scores[mask].reshape(
                    l_scores.size(0), -1)
                diff = l_scores[:, None] - l_other_scores
                diff.clamp_(min=1e-7, max=self.max_diff)
                loss_base_margin.append((1 - l_scores[:, None]).pow(self.power_weight) * -diff.log())
            loss_base_margin = torch.cat(loss_base_margin)
            if self.reduction == 'sum':
                loss_base_margin = loss_base_margin.sum().div(avg_factor)
            else:
                loss_base_margin = loss_base_margin.mean()
            loss_base_margin *= loss_base_margin_weight
        else:
            loss_base_margin = cls_score.sum() * 0.

        losses['loss_base_margin'] = loss_base_margin

        # novel margin
        num_novel = novel_scores.size(0)
        if num_novel > 0:
            loss_novel_margin = []
            # loss_novel_margin2 = []
            for label in novel_labels.unique():
                l_inds = labels == label
                l_scores = scores[l_inds, label]
                l_all_scores = scores[l_inds]
                mask = torch.ones_like(l_all_scores).bool()
                mask[:, label] = False
                if self.ignore_neg:
                    mask[:, -1] = False
                l_other_scores = l_all_scores[mask].reshape(
                    l_scores.size(0), -1)
                diff = l_scores[:, None] - l_other_scores
                diff.clamp_(min=1e-7, max=self.max_diff)  # 小于self.max_diff产生loss
                loss_novel_margin.append((1 - l_scores[:, None]).pow(self.power_weight) * -diff.log())
            loss_novel_margin = torch.cat(loss_novel_margin)
            if self.reduction == 'sum':
                loss_novel_margin = loss_novel_margin.sum().div(avg_factor)
            else:
                loss_novel_margin = loss_novel_margin.mean()
            loss_novel_margin *= loss_novel_margin_weight
        else:
            loss_novel_margin = cls_score.sum() * 0.
        losses['loss_novel_margin'] = loss_novel_margin

        # neg margin
        neg_inds = labels == self.num_classes
        neg_scores = scores[neg_inds, -1]
        neg_other_scores = scores[neg_inds, :-1]
        diff = neg_scores[:, None] - neg_other_scores
        diff.clamp_(min=1e-7, max=self.max_diff)
        loss_neg_margin = (1 - neg_scores[:, None]).pow(self.power_weight) * -diff.log()

        if self.reduction == 'sum':
            loss_neg_margin = loss_neg_margin.sum().div(avg_factor)
        else:
            loss_neg_margin = loss_neg_margin.mean()
        loss_neg_margin *= loss_neg_margin_weight
        losses['loss_neg_margin'] = loss_neg_margin

        if self.reduction == 'sum':
            if 'loss_novel_margin' in losses:
                losses['loss_novel_margin'].clamp_(max=self.max_loss)
            if 'loss_base_margin' in losses:
                losses['loss_base_margin'].clamp_(max=self.max_loss)
            if 'loss_neg_margin' in losses:
                losses['loss_neg_margin'].clamp_(max=self.max_loss)

        return losses


@LOSSES.register_module()
class AdaptiveDisLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 shot,
                 base_class_count=None,
                 loss_base_margin_weight=None,
                 loss_novel_margin_weight=None,
                 loss_neg_margin_weight=0.001,
                 max_diff=1.0,
                 max_loss=1.0,
                 ignore_neg=False,
                 reduction='mean',
                 loss_weight=1.0,
                 power_weight=2.0,
                 min_samples_per_class=2,
                 eps_diff=1e-3,
                 max_log=5.0):
        """
        改进版 DisLoss：
        - reduction 默认 'mean'（对 small-batch 更稳健）
        - 增加 base_class_count 显式参数（避免硬编码）
        - min_samples_per_class: 避免单样本类导致统计噪声
        - eps_diff / max_log: 数值保护，避免 -log(diff) 爆发
        """
        super(AdaptiveDisLoss, self).__init__()
        self.max_diff = max_diff
        self.max_loss = max_loss
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.power_weight = power_weight
        self.min_samples_per_class = min_samples_per_class

        # 权重初始化（保持和原先逻辑兼容）
        if loss_novel_margin_weight is None:
            self.loss_novel_margin_weight = 1. / max(shot, 1)
        else:
            self.loss_novel_margin_weight = loss_novel_margin_weight

        if loss_base_margin_weight is None:
            self.loss_base_margin_weight = self.loss_novel_margin_weight / 3.
        else:
            self.loss_base_margin_weight = loss_base_margin_weight

        self.loss_neg_margin_weight = loss_neg_margin_weight

        self.num_classes = num_classes
        self.base_class_count = base_class_count
        self.ignore_neg = ignore_neg

        # 数值保护参数
        self.eps_diff = eps_diff
        self.max_log = max_log

    def forward(self, cls_score, labels, label_weights=None, decay_rate=None, **kwargs):
        """
        cls_score: raw logits (未 softmax)
        labels: long tensor (N,)
        label_weights: optional mask tensor (N,) where >0 means valid
        decay_rate: optional warmup scalar in [0,1]
        """
        losses = dict()
        device = cls_score.device

        # apply decay_rate if provided (warmup)
        loss_base_margin_weight = self.loss_base_margin_weight
        loss_novel_margin_weight = self.loss_novel_margin_weight
        loss_neg_margin_weight = self.loss_neg_margin_weight
        if decay_rate is not None:
            loss_base_margin_weight = loss_base_margin_weight * decay_rate
            loss_novel_margin_weight = loss_novel_margin_weight * decay_rate
            loss_neg_margin_weight = loss_neg_margin_weight * decay_rate

        # base_class_count fallback
        if self.base_class_count is None:
            # legacy behavior: 3/4 of classes as base
            num_base_classes = self.num_classes // 4 * 3
        else:
            num_base_classes = self.base_class_count

        # softmax -> probabilities
        probs = torch.softmax(cls_score, dim=-1)

        # label mask / avg factor
        if label_weights is None:
            valid_mask = torch.ones_like(labels).bool()
        else:
            valid_mask = label_weights > 0

        avg_factor = max(valid_mask.sum().float().item(), 1.0)

        # splits
        base_inds = valid_mask & (labels < num_base_classes)
        novel_inds = valid_mask & (labels >= num_base_classes) & (labels < self.num_classes)

        base_labels = labels[base_inds]
        novel_labels = labels[novel_inds]

        # helper: safe clamp for (1 - p)
        def safe_one_minus(x):
            return torch.clamp(1.0 - x, min=1e-4, max=1.0)

        # ========== base margin ==========
        loss_base_margin = torch.tensor(0., device=device)
        if base_inds.any():
            per_class_losses = []
            for lbl in torch.unique(base_labels):
                lbl = int(lbl.item())
                inds = (labels == lbl) & valid_mask
                if inds.sum() < self.min_samples_per_class:
                    # skip classes with too few samples to avoid noisy estimates
                    continue
                l_scores = probs[inds, lbl].view(-1, 1)  # (n,1)
                l_all = probs[inds]  # (n, C)
                # mask out the current label
                mask = torch.ones_like(l_all).bool()
                mask[:, lbl] = False
                if self.ignore_neg:
                    mask[:, -1] = False
                other = l_all[mask].view(l_scores.size(0), -1)  # (n, C-1)
                diff = (l_scores - other).clamp(min=self.eps_diff, max=self.max_diff)  # (n, C-1)
                # stable -log
                log_term = torch.clamp(-torch.log(diff), max=self.max_log)
                # weighting term
                weight_term = safe_one_minus(l_scores).pow(self.power_weight)
                # per-sample-per-other loss
                per = weight_term * log_term  # shape (n, C-1)
                # aggregate per-class
                per_class_losses.append(per.view(-1))
            if len(per_class_losses) > 0:
                loss_base_margin = torch.cat(per_class_losses).mean()  # mean over elements
                loss_base_margin = loss_base_margin * loss_base_margin_weight
        losses['loss_base_margin'] = loss_base_margin

        # ========== novel margin ==========
        loss_novel_margin = torch.tensor(0., device=device)
        if novel_inds.any():
            per_class_losses = []
            for lbl in torch.unique(novel_labels):
                lbl = int(lbl.item())
                inds = (labels == lbl) & valid_mask
                if inds.sum() < self.min_samples_per_class:
                    continue
                l_scores = probs[inds, lbl].view(-1, 1)
                l_all = probs[inds]
                mask = torch.ones_like(l_all).bool()
                mask[:, lbl] = False
                if self.ignore_neg:
                    mask[:, -1] = False
                other = l_all[mask].view(l_scores.size(0), -1)
                diff = (l_scores - other).clamp(min=self.eps_diff, max=self.max_diff)
                log_term = torch.clamp(-torch.log(diff), max=self.max_log)
                weight_term = safe_one_minus(l_scores).pow(self.power_weight)
                per = weight_term * log_term
                per_class_losses.append(per.view(-1))
            if len(per_class_losses) > 0:
                loss_novel_margin = torch.cat(per_class_losses).mean()
                loss_novel_margin = loss_novel_margin * loss_novel_margin_weight
        losses['loss_novel_margin'] = loss_novel_margin

        # ========== neg / background margin ==========
        # neg_inds = labels == self.num_classes  (assumes background label == num_classes)
        neg_mask = valid_mask & (labels == self.num_classes)
        if neg_mask.any():
            neg_scores = probs[neg_mask, -1].view(-1, 1)
            neg_other = probs[neg_mask, :-1]
            diff = (neg_scores - neg_other).clamp(min=self.eps_diff, max=self.max_diff)
            log_term = torch.clamp(-torch.log(diff), max=self.max_log)
            weight_term = safe_one_minus(neg_scores).pow(self.power_weight)
            neg_per = weight_term * log_term
            if neg_per.numel() > 0:
                if self.reduction == 'sum':
                    loss_neg_margin = neg_per.sum().div(avg_factor)
                else:
                    loss_neg_margin = neg_per.mean()
                loss_neg_margin = loss_neg_margin * loss_neg_margin_weight
            else:
                loss_neg_margin = torch.tensor(0., device=device)
        else:
            loss_neg_margin = torch.tensor(0., device=device)
        losses['loss_neg_margin'] = loss_neg_margin

        # clamp overall (对 mean 也进行保护)
        for k in ['loss_base_margin', 'loss_novel_margin', 'loss_neg_margin']:
            if k in losses:
                losses[k] = torch.clamp(losses[k], max=self.max_loss)

        # multiply global loss_weight if needed (外层 DualMargin head 处也会有 dis_loss_weight)
        return losses
