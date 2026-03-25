import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from mmdet.models.losses import accuracy
from mmcv.ops.nms import batched_nms


@HEADS.register_module()
class IncreaseBBoxHead(ConvFCBBoxHead):

    def __init__(self,
                 fc_out_channels=1024,
                 scale=20.,
                 base_cpt=None,
                 base_alpha=0.5,
                 *args,
                 **kwargs):
        super(IncreaseBBoxHead,
              self).__init__(*args,
                             **kwargs)
        # del self.fc_cls
        del self.shared_fcs
        del self.cls_fcs
        del self.reg_fcs
        assert base_cpt is not None
        self.base_cpt = base_cpt
        self.base_alpha = base_alpha

        num_base = self.num_classes // 4 * 3
        num_novel = self.num_classes // 4

        # base branch
        base_shared_fcs = nn.ModuleList()
        base_shared_fcs.append(nn.Linear(49 * 256, fc_out_channels))
        base_shared_fcs.append(nn.Linear(fc_out_channels, fc_out_channels))
        self.base_shared_fcs = base_shared_fcs
        # self.base_fc_cls = nn.Linear(self.cls_last_dim, num_base, bias=False)

        # novel branch
        novel_shared_fcs = nn.ModuleList()
        novel_shared_fcs.append(nn.Linear(49 * 256, fc_out_channels))
        novel_shared_fcs.append(nn.Linear(fc_out_channels, fc_out_channels))
        self.novel_shared_fcs = novel_shared_fcs

        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # temperature
        self.scale = scale

        print(base_shared_fcs)

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

    def forward(self, x, return_fc_feat=False):

        x = x.flatten(1)

        alpha = self.base_alpha

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)

        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1 - alpha) * novel_x
            x = self.relu(x)

        bbox_preds = self.fc_reg(x)

        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                                   dim=1).unsqueeze(1).expand_as(
                self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds


@HEADS.register_module()
class KDBBoxHead(IncreaseBBoxHead):
    def __init__(self,
                 loss_kd_weight=0.001,
                 base_alpha=0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.loss_kd = dict()
        self.loss_kd_weight = loss_kd_weight
        self.base_alpha = base_alpha

    def forward(self, x, return_fc_feat=False):

        loss_feature_kd = 0

        x = x.flatten(1)
        alpha = self.base_alpha

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)

        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1 - alpha) * novel_x
            if self.training:
                loss_feature_kd += torch.dist(alpha * base_x + (1 - alpha) * novel_x, base_x, 2)

            x = self.relu(x)

        if self.training:
            loss_kd = loss_feature_kd / 2.0 * self.loss_kd_weight
            self.loss_kd['loss_kd'] = loss_kd

        bbox_preds = self.fc_reg(x)
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                                   dim=1).unsqueeze(1).expand_as(
                self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
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

        if cls_score is not None:
            losses.update(self.loss_kd)

        return losses


class AttentionFusion2D(nn.Module):
    def __init__(self, size=1024):
        super(AttentionFusion2D, self).__init__()
        self.attn = nn.Linear(size, size)  # 线性变换计算注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, f1, f2):
        w1 = self.sigmoid(self.attn(f1))  # 计算 F1 的注意力权重
        w2 = self.sigmoid(self.attn(f2))  # 计算 F2 的注意力权重

        # 归一化权重
        w1 = w1 / (w1 + w2 + 1e-6)
        w2 = w2 / (w1 + w2 + 1e-6)

        F_fused = w1 * f1 + w2 * f2  # 加权求和
        return F_fused


class ContrastiveAttentionFusion(nn.Module):
    def __init__(self, size=1024):
        super(ContrastiveAttentionFusion, self).__init__()
        self.attn = nn.Linear(size, size)  # 计算注意力
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2):
        # 计算特征相似性（余弦相似度）
        cos_sim = F.cosine_similarity(F1, F2, dim=-1, eps=1e-8)  # (1024,)

        # 计算 F1 和 F2 的注意力权重
        w1 = self.sigmoid(self.attn(F1)) * (1 - cos_sim.unsqueeze(-1))  # 1-相似度 -> 去冗余
        w2 = self.sigmoid(self.attn(F2)) * cos_sim.unsqueeze(-1)

        # 归一化权重
        w1 = w1 / (w1 + w2 + 1e-6)
        w2 = w2 / (w1 + w2 + 1e-6)

        # 进行加权融合
        F_fused = w1 * F1 + w2 * F2
        return F_fused


class BaseSharedKnowledgeTransferFusion(nn.Module):
    def __init__(self, size=1024):
        super(BaseSharedKnowledgeTransferFusion, self).__init__()
        self.linear = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, base_x, novel_x):
        # 计算相似性权重，衡量base_x和novel_x的共享部分
        similarity = F.cosine_similarity(base_x, novel_x, dim=-1, eps=1e-6)
        similarity = similarity.unsqueeze(-1)  # (1024, 1)

        # 计算注意力权重
        w_base = self.sigmoid(self.linear(base_x))
        w_novel = self.sigmoid(self.linear(novel_x))

        # **核心调整：增强相似部分，减少不同部分的距离**，强调base_x类的共享特征
        knowledge_transfer = similarity * base_x + (1 - similarity) * novel_x

        # 结合注意力权重进行加权融合
        w_base = w_base / (w_base + w_novel + 1e-6)
        w_novel = w_novel / (w_base + w_novel + 1e-6)

        F_fused = w_base * knowledge_transfer + w_novel * novel_x
        return F_fused


class NovelSharedKnowledgeTransferFusion(nn.Module):
    def __init__(self, size=1024):
        super(NovelSharedKnowledgeTransferFusion, self).__init__()
        self.linear = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, base_x, novel_x):
        # 计算相似性权重，衡量base_x和novel_x的共享部分
        similarity = F.cosine_similarity(base_x, novel_x, dim=-1, eps=1e-6)
        similarity = similarity.unsqueeze(-1)  # (1024, 1)

        # 计算注意力权重
        w_base = self.sigmoid(self.linear(base_x))
        w_novel = self.sigmoid(self.linear(novel_x))

        # **核心调整：增强相似部分，减少不同部分的距离**, 强调novel_x类的共享特征
        knowledge_transfer = similarity * novel_x + (1 - similarity) * base_x

        # 结合注意力权重进行加权融合
        w_base = w_base / (w_base + w_novel + 1e-6)
        w_novel = w_novel / (w_base + w_novel + 1e-6)

        F_fused = w_base * knowledge_transfer + w_novel * novel_x
        return F_fused


class BaseKnowledgeTransferFusion(nn.Module):
    def __init__(self, size=1024):
        super(BaseKnowledgeTransferFusion, self).__init__()
        self.linear = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, base_x, novel_x):
        # 计算相似性权重，衡量base_x和novel_x的共享部分
        similarity = F.cosine_similarity(base_x, novel_x, dim=-1, eps=1e-6)
        similarity = similarity.unsqueeze(-1)  # (1024, 1)

        # 计算注意力权重
        w_base = self.sigmoid(self.linear(base_x))
        w_novel = self.sigmoid(self.linear(novel_x))

        # **核心调整：增强相似部分，减少不同部分的距离**, base_x类的引导
        knowledge_transfer = base_x + (1 - similarity) * novel_x

        # 结合注意力权重进行加权融合
        w_base = w_base / (w_base + w_novel + 1e-6)
        w_novel = w_novel / (w_base + w_novel + 1e-6)

        F_fused = w_base * knowledge_transfer + w_novel * novel_x
        return F_fused


class NovelKnowledgeTransferFusion(nn.Module):
    def __init__(self, size=1024):
        super(NovelKnowledgeTransferFusion, self).__init__()
        self.linear = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, base_x, novel_x):
        # 计算相似性权重，衡量base_x和novel_x的共享部分
        similarity = F.cosine_similarity(base_x, novel_x, dim=-1, eps=1e-6)
        similarity = similarity.unsqueeze(-1)  # (1024, 1)

        # 计算注意力权重
        w_base = self.sigmoid(self.linear(base_x))
        w_novel = self.sigmoid(self.linear(novel_x))

        # **核心调整：增强相似部分，减少不同部分的距离**, novel_x类的引导
        knowledge_transfer = novel_x + (1 - similarity) * base_x

        # 结合注意力权重进行加权融合
        w_base = w_base / (w_base + w_novel + 1e-6)
        w_novel = w_novel / (w_base + w_novel + 1e-6)

        F_fused = w_base * knowledge_transfer + w_novel * novel_x
        return F_fused


class FeatureBlender(nn.Module):
    def __init__(self, alpha=0.5, learnable=False):
        """
        alpha: 初始混合系数
        learnable: 如果为 True，alpha 将作为可学习参数
        """
        super(FeatureBlender, self).__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.register_buffer('alpha', torch.tensor(alpha))

    def forward(self, base_x, novel_x):
        """
        按 alpha 融合 base_x 和 novel_x
        """
        return self.alpha * base_x + (1 - self.alpha) * novel_x


@HEADS.register_module()
class DisKDBBoxHead(KDBBoxHead):

    def __init__(self,
                 dis_loss=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if dis_loss is not None:
            self.dis_loss = build_loss(copy.deepcopy(dis_loss))
        else:
            self.dis_loss = None

    def forward(self, x, return_fc_feat=False):

        kd_loss_list = []
        loss_feature_kd = 0
        x = x.flatten(1)
        alpha = self.base_alpha
        base_x = x
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

        bbox_preds = self.fc_reg(x)
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                                   dim=1).unsqueeze(1).expand_as(
                self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds

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

        if cls_score is not None:
            losses.update(self.loss_kd)

        if self.dis_loss is not None and cls_score is not None:
            losses.update(self.dis_loss(cls_score, labels, label_weights))

        return losses


@HEADS.register_module()
class OurDisKDBBoxHead(DisKDBBoxHead):

    def __init__(self,
                 dis_loss=None,
                 fusion_type=None,
                 kd_T=4.0,
                 kd_alpha=0.5,
                 lambda_base=0.1,
                 lambda_novel=10,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if dis_loss is not None:
            self.dis_loss = build_loss(copy.deepcopy(dis_loss))
        else:
            self.dis_loss = None
        self.kd_T = kd_T  # 温控因子
        self.kd_alpha = kd_alpha  # 双向蒸馏权重
        self.lambda_base = lambda_base
        self.lambda_novel = lambda_novel

        self.fusion_type = fusion_type

        if fusion_type == 'BaseKnowledgeTransferFusion':
            self.fusion_layer = BaseKnowledgeTransferFusion(1024)
        elif fusion_type == 'NovelKnowledgeTransferFusion':
            self.fusion_layer = NovelKnowledgeTransferFusion(1024)
        elif fusion_type == 'ContrastiveAttentionFusion':
            self.fusion_layer = ContrastiveAttentionFusion(1024)
        elif fusion_type == 'AttentionFusion2D':
            self.fusion_layer = AttentionFusion2D(1024)
        elif fusion_type == 'BaseSharedKnowledgeTransferFusion':
            self.fusion_layer = BaseSharedKnowledgeTransferFusion(1024)
        elif fusion_type == 'NovelSharedKnowledgeTransferFusion':
            self.fusion_layer = NovelSharedKnowledgeTransferFusion(1024)
        else:
            self.fusion_layer = FeatureBlender()

    def kd_loss_bidirectional(self, base_x, novel_x):
        """
        双向蒸馏 + 温控因子 + feature-level Frobenius norm
        """
        T = self.kd_T
        alpha = self.kd_alpha
        # --- feature-level 蒸馏 ---
        feat_loss = torch.norm(base_x - novel_x, p='fro', dim=-1).mean()

        # --- logit-level 蒸馏 ---
        # 使用温控后的 softmax 分布
        p_log = F.log_softmax(base_x / T, dim=-1)
        q_soft = F.softmax(novel_x / T, dim=-1)
        kd_forward = F.kl_div(p_log, q_soft, reduction='batchmean') * (T * T)

        q_log = F.log_softmax(novel_x / T, dim=-1)
        p_soft = F.softmax(base_x / T, dim=-1)
        kd_backward = F.kl_div(q_log, p_soft, reduction='batchmean') * (T * T)

        # --- 双向组合 ---
        kd_bi = alpha * kd_forward + (1 - alpha) * kd_backward

        return feat_loss + kd_bi

    def forward(self, x, return_fc_feat=False):

        kd_loss_list = []
        loss_feature_kd = 0
        x_in = x.flatten(1)  # 将第一位及其的特征进行展平 (2, 3, 4) -> (2, 12)
        alpha = self.base_alpha
        base_x = x_in
        x = x_in
        # print(f"输入x_in尺度为：{x_in.shape}")    # (1000, 12544)

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)

            # # ---- 构建对比学习的正负样本 ----
            # contrast_loss = contrastive_loss(base_x, novel_x, temperature=0.2)

            ###################### 使用自注意力机制进行融合两者 #############################
            # if not self.fusion_type:
            x_fuse = self.fusion_layer(base_x, novel_x)

            # -------- Contrastive Learning --------
            loss_pos1 = contrastive_loss(base_x, novel_x, temperature=0.2)
            loss_pos2 = contrastive_loss(base_x, x_fuse.clone(), temperature=0.2)
            loss_pos3 = contrastive_loss(novel_x, x_fuse.clone(), temperature=0.2)
            contrast_loss = (loss_pos1 + loss_pos2 + loss_pos3) / 3.0

            # else:
            #     print("使用原始的加权平均来进行学习")
            #     x_fuse = alpha * base_x + (1 - alpha) * novel_x  # 修改，这部分可以修改成一个self_attention模块
            ############################################################################
            # kd_loss_list.append(torch.frobenius_norm(x - novel_x, dim=-1))        # 按照矩阵行计算每个元素的平方和在开方
            # kd_loss_list.append(self.kd_loss_bidirectional(base_x, novel_x))

            # print(f"输入base_x尺度为：{base_x.shape}")    # (1000, 1024)
            # print(f"输入novel_x尺度为：{novel_x.shape}")  # (1000, 1024)
            # print(f"输入x_fuse尺度为：{x_fuse.shape}")    # (1000, 1024)
            # ---------- 蒸馏损失 ----------
            # 1) Base 避免遗忘：蒸馏输入 x 和 base_x
            # if not proj_layer:
            #     self.proj_layer = nn.Linear(x_in.size(1), base_x.size(1)).to(x_in.device)
            #     proj_layer = True
            #     x_in_proj = self.proj_layer(x_in)  # (1000, 1024)
            # x_in = x_in_proj
            # kd_loss_base = self.kd_loss_bidirectional(base_x, x_fuse)
            # kd_loss_base = torch.frobenius_norm(x - base_x, dim=-1)
            kd_loss_base = 0.0

            # 2) Novel 学习补偿：蒸馏 novel_x 和融合特征 x_fuse
            kd_loss_novel = self.kd_loss_bidirectional(novel_x, x_fuse)
            # kd_loss_novel = torch.frobenius_norm(novel_x-x_fuse, dim=-1)

            kd_loss_list.append(self.lambda_base * kd_loss_base + self.lambda_novel * kd_loss_novel)

            x = self.relu(x_fuse)
        # kd_loss = torch.cat(kd_loss_list, dim=0)
        # kd_loss = torch.mean(kd_loss)
        kd_loss = torch.stack(kd_loss_list).mean()

        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] = kd_loss
            self.loss_kd['loss_contrast'] = self.loss_kd_weight * contrast_loss  # 新增对比学习损失

        bbox_preds = self.fc_reg(x)
        # 计算余弦相似度进行分类，先将特征进行归一化，然后将fc_cls分类器的权重也归一化，最后计算两者的余弦相似度
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                                   dim=1).unsqueeze(1).expand_as(
                self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds

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

        if cls_score is not None:
            losses.update(self.loss_kd)

        if self.dis_loss is not None and cls_score is not None:
            losses.update(self.dis_loss(cls_score, labels, label_weights))

        return losses


@HEADS.register_module()
class ChangeDisKDBBoxHead(DisKDBBoxHead):

    def __init__(self,
                 dis_loss=None,
                 fusion_type=None,
                 loss_mi_weight=0.025,
                 *args,
                 **kwargs):
        super(ChangeDisKDBBoxHead, self).__init__(*args, **kwargs)
        if dis_loss is not None:
            self.dis_loss = build_loss(copy.deepcopy(dis_loss))
        else:
            self.dis_loss = None

        self.fusion_type = fusion_type
        self.loss_mi_weight = loss_mi_weight

        if fusion_type == 'BaseKnowledgeTransferFusion':
            self.fusion_layer = BaseKnowledgeTransferFusion(1024)
        elif fusion_type == 'NovelKnowledgeTransferFusion':
            self.fusion_layer = NovelKnowledgeTransferFusion(1024)
        elif fusion_type == 'ContrastiveAttentionFusion':
            self.fusion_layer = ContrastiveAttentionFusion(1024)
        elif fusion_type == 'AttentionFusion2D':
            self.fusion_layer = AttentionFusion2D(1024)
        elif fusion_type == 'BaseSharedKnowledgeTransferFusion':
            self.fusion_layer = BaseSharedKnowledgeTransferFusion(1024)
        elif fusion_type == 'NovelSharedKnowledgeTransferFusion':
            self.fusion_layer = NovelSharedKnowledgeTransferFusion(1024)
        else:
            self.fusion_layer = FeatureBlender(self.base_alpha)

    def relation_distillation_loss(self, base_feat, novel_feat, scale=5.0, reduction='mean'):
        """
        Cross-Branch Relation Distillation (Relation KD)
        ------------------------------------------------
        保持 novel_feat 与 base_feat 的结构相似性，而非直接对齐特征值。
        支持 Base 与 Novel 样本数量不一致。

        Args:
            base_feat (Tensor): shape [N_b, C], 来自 Base 分支
            novel_feat (Tensor): shape [N_n, C], 来自 Novel 分支
            scale (float): 相似度缩放因子，默认 5.0
            reduction (str): {'mean', 'sum', 'none'}

        Returns:
            Tensor: 计算得到的关系蒸馏损失 (标量)
        """
        # ---- Step 1. 特征归一化 ----
        base_norm = F.normalize(base_feat, p=2, dim=1)
        novel_norm = F.normalize(novel_feat, p=2, dim=1)

        # ---- Step 2. 计算两组样本的结构相似度 ----
        # 注意：Base 与 Novel 数量可能不同，所以使用跨集合相似度矩阵
        sim_base = torch.mm(base_norm, base_norm.t())  # [N_b, N_b]
        sim_cross = torch.mm(novel_norm, base_norm.t())  # [N_n, N_b]

        # ---- Step 3. 对齐结构：对每个 Novel 样本，使其与 Base 的相似分布一致 ----
        # 使用 softmax 归一化分布形式的KL散度（结构对齐）
        p_base = F.softmax(scale * sim_base.detach(), dim=-1)
        p_novel = F.log_softmax(scale * sim_cross, dim=-1)

        kd_loss = F.kl_div(p_novel, p_base.expand_as(p_novel), reduction='none').sum(dim=-1)

        if reduction == 'mean':
            kd_loss = kd_loss.mean()
        elif reduction == 'sum':
            kd_loss = kd_loss.sum()

        return kd_loss

    # ===============================
    # MI LOSS between base_x & novel_x
    # ===============================
    def mutual_information_loss(self, base_x, novel_x, tau=0.07):
        # 对齐样本数
        N_b, N_n = base_x.size(0), novel_x.size(0)
        if N_b != N_n:
            if N_b > N_n:
                idx = torch.randperm(N_b)[:N_n]
                base_x = base_x[idx]
            else:
                idx = torch.randperm(N_n)[:N_b]
                novel_x = novel_x[idx]

        # 归一化特征
        base_norm = F.normalize(base_x, dim=1)
        novel_norm = F.normalize(novel_x, dim=1)

        # 相似度矩阵
        sim_matrix = torch.matmul(novel_norm, base_norm.t()) / tau  # [N, N]
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()  # 数值稳定性

        exp_sim = torch.exp(sim_matrix)
        pos_sim = torch.diag(exp_sim)

        # === 关键修改部分 ===
        # 我们要最小化 mutual information，即鼓励相似度降低
        # 所以直接取相似度的均值作为损失（越大越相似，越小越好）
        mi_loss = torch.mean(torch.log(exp_sim.sum(dim=1) / (pos_sim + 1e-8)))

        return mi_loss

    def forward(self, x, return_fc_feat=False):

        kd_loss_list = []
        loss_feature_kd = 0
        x = x.flatten(1)
        alpha = self.base_alpha
        base_x = x
        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = self.fusion_layer(base_x, novel_x)
            kd_loss_list.append(torch.frobenius_norm(base_x - x, dim=-1))
            # kd_loss_list.append(self.relation_distillation_loss(base_x.detach(), novel_x.detach(), scale=5.0))
            x = self.relu(x)
        # kd_loss = torch.stack(kd_loss_list)
        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)
        mi_loss = self.mutual_information_loss(base_x.detach(), novel_x)

        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            mi_loss = mi_loss * self.loss_mi_weight
            self.loss_kd['loss_kd'] = kd_loss
            self.loss_kd['loss_mi'] = mi_loss

        bbox_preds = self.fc_reg(x)
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                                   dim=1).unsqueeze(1).expand_as(
                self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds


def contrastive_loss(feat_q, feat_k, temperature=0.2):
    """feat_q: [N, D], feat_k: [N, D]"""
    # 归一化
    feat_q = F.normalize(feat_q, dim=1)
    feat_k = F.normalize(feat_k, dim=1)

    # 相似度矩阵 [N, N]
    logits = torch.mm(feat_q, feat_k.t()) / temperature

    # 正样本对是对角线
    labels = torch.arange(feat_q.size(0)).long().to(feat_q.device)

    loss = F.cross_entropy(logits, labels)
    return loss


@HEADS.register_module()
class OldArcFaceDisKDBBoxHead(DisKDBBoxHead):
    # class ArcFaceDisKDBBoxHead(ConvFCBBoxHead):
    """DisKDBBoxHead with ArcFaceLoss instead of CrossEntropyLoss."""

    def __init__(self,
                 arcface_loss_cfg=dict(
                     type='ArcFaceLoss',
                     s=64.0,
                     margin=0.5,
                     loss_weight=1.0,
                     reduction='mean'),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # 构造 ArcFaceLoss
        self.loss_cls = build_loss(arcface_loss_cfg)

    def forward(self, x, return_fc_feat=False):
        """和 DisKDBBoxHead 一样，不改 forward，只是替换 loss"""
        return super().forward(x, return_fc_feat)
        # return super().forward(x)

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
        """替换 CE Loss 为 ArcFaceLoss"""
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                # 用 ArcFaceLoss 计算分类损失
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    weight=label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                # acc 还是用原始 logits
                from mmdet.models.losses.accuracy import accuracy
                losses['acc'] = accuracy(cls_score, labels)

        # bbox 回归部分保持不变
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1, 4)[
                        pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()

        # KD 损失
        if cls_score is not None:
            losses.update(self.loss_kd)

        # DisLoss（如果有的话）
        if self.dis_loss is not None and cls_score is not None:
            losses.update(self.dis_loss(cls_score, labels, label_weights))

        return losses


@HEADS.register_module()
class FusionDisKDBBoxHead(DisKDBBoxHead):
    """DisKDBBoxHead with feature fusion (configurable fusion layer)."""

    def __init__(self,
                 fusion_type=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.fusion_type = fusion_type
        if fusion_type == 'BaseKnowledgeTransferFusion':
            self.fusion_layer = BaseKnowledgeTransferFusion(1024)
        elif fusion_type == 'NovelKnowledgeTransferFusion':
            self.fusion_layer = NovelKnowledgeTransferFusion(1024)
        elif fusion_type == 'ContrastiveAttentionFusion':
            self.fusion_layer = ContrastiveAttentionFusion(1024)
        elif fusion_type == 'AttentionFusion2D':
            self.fusion_layer = AttentionFusion2D(1024)
        elif fusion_type == 'BaseSharedKnowledgeTransferFusion':
            self.fusion_layer = BaseSharedKnowledgeTransferFusion(1024)
        elif fusion_type == 'NovelSharedKnowledgeTransferFusion':
            self.fusion_layer = NovelSharedKnowledgeTransferFusion(1024)
        else:
            self.fusion_layer = FeatureBlender()

    def forward(self, x, return_fc_feat=False):

        kd_loss_list = []
        loss_feature_kd = 0
        x = x.flatten(1)  # 将第一位及其的特征进行展平 (2, 3, 4) -> (2, 12)
        alpha = self.base_alpha
        base_x = x
        # print(f"输入x_in尺度为：{x_in.shape}")    # (1000, 12544)

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)

            x = self.fusion_layer(base_x, novel_x)

            kd_loss_list.append(torch.frobenius_norm(base_x - x, dim=-1))  # 按照矩阵行计算每个元素的平方和在开方

            x = self.relu(x)
        kd_loss = torch.cat(kd_loss_list, dim=0)
        kd_loss = torch.mean(kd_loss)

        if self.training:
            kd_loss = kd_loss * self.loss_kd_weight
            self.loss_kd['loss_kd'] = kd_loss

        bbox_preds = self.fc_reg(x)
        # 计算余弦相似度进行分类，先将特征进行归一化，然后将fc_cls分类器的权重也归一化，最后计算两者的余弦相似度
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                                   dim=1).unsqueeze(1).expand_as(
                self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds


@HEADS.register_module()
class ContrastiveDisKDBBoxHead(FusionDisKDBBoxHead):
    """DisKDBBoxHead with configurable KD/KDB/Contrastive losses."""

    def __init__(self,
                 use_kd=True,
                 use_kdb=False,
                 use_contrastive=False,
                 kdb_cfg=dict(
                     loss_weight=0.025,
                     kd_T=4.0,
                     kd_alpha=0.5,
                     lambda_base=0.1,
                     lambda_novel=10,
                 ),
                 contrastive_cfg=dict(
                     temperature=0.2,
                     loss_weight=0.5,
                     pos1_weight=1.0,
                     pos2_weight=1.0,
                     pos3_weight=1.0),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # 开关
        self.use_kd = use_kd
        self.use_kdb = use_kdb
        self.use_contrastive = use_contrastive

        # 定义损失
        self.loss_kd = dict()
        self.loss_contrast = dict()
        self.loss_kdb = dict()

        # KD / KDB 权重
        self.loss_kdb_weight = kdb_cfg.get("loss_weight", 1.0)
        self.kd_T = kdb_cfg.get("kd_T", 1.0)
        self.kd_alpha = kdb_cfg.get("kd_alpha", 1.0)
        self.lambda_base = kdb_cfg.get("lambda_base", 1.0)
        self.lambda_novel = kdb_cfg.get("lambda_novel", 1.0)

        # 对比学习配置
        self.temperature = contrastive_cfg.get("temperature", 0.2)
        self.contrastive_loss_weight = contrastive_cfg.get("loss_weight", 1.0)
        self.pos1_weight = contrastive_cfg.get("pos1_weight", 1.0)
        self.pos2_weight = contrastive_cfg.get("pos2_weight", 1.0)
        self.pos3_weight = contrastive_cfg.get("pos3_weight", 1.0)

    def kd_loss_bidirectional(self, base_x, novel_x):
        """
        双向蒸馏 + 温控因子 + feature-level Frobenius norm
        """
        T = self.kd_T
        alpha = self.kd_alpha

        # --- feature-level 蒸馏 ---
        feat_loss = torch.norm(base_x - novel_x, p='fro', dim=-1).mean()

        # --- logit-level 蒸馏 ---
        # 使用温控后的 softmax 分布
        p_log = F.log_softmax(base_x / T, dim=-1)
        q_soft = F.softmax(novel_x / T, dim=-1)
        kd_forward = F.kl_div(p_log, q_soft, reduction='batchmean') * (T * T)

        q_log = F.log_softmax(novel_x / T, dim=-1)
        p_soft = F.softmax(base_x / T, dim=-1)
        kd_backward = F.kl_div(q_log, p_soft, reduction='batchmean') * (T * T)

        # --- 双向组合 ---
        kd_bi = alpha * kd_forward + (1 - alpha) * kd_backward

        return feat_loss + kd_bi

    def forward(self, x, return_fc_feat=False):
        kd_loss_list, kdb_loss_list, contrastive_loss_list = [], [], []

        x = x.flatten(1)
        alpha = self.base_alpha
        base_x = x

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)

            # -------- 特征融合 --------
            x_fuse = self.fusion_layer(base_x, novel_x)

            # -------- 对比损失 --------
            if self.use_contrastive:
                contrast_loss = 0.0
                if self.pos1_weight > 0:
                    loss_pos1 = contrastive_loss(base_x, novel_x, temperature=self.temperature)
                    contrast_loss += self.pos1_weight * loss_pos1
                if self.pos2_weight > 0:
                    loss_pos2 = contrastive_loss(base_x, x_fuse.clone(), temperature=self.temperature)
                    contrast_loss += self.pos2_weight * loss_pos2
                if self.pos3_weight > 0:
                    loss_pos3 = contrastive_loss(novel_x, x_fuse.clone(), temperature=self.temperature)
                    contrast_loss += self.pos3_weight * loss_pos3
                contrastive_loss_list.append(contrast_loss)

            # -------- 单向 KD 损失 --------
            if self.use_kd:
                kd_loss_list.append((torch.frobenius_norm(base_x - x_fuse, dim=-1) +
                                    torch.frobenius_norm(base_x - novel_x, dim=-1))/2)

            # -------- 双向 KD 损失 --------
            if self.use_kdb:
                kdb_loss_list.append(self.kd_loss_bidirectional(base_x, x_fuse))

            x = self.relu(x_fuse)

        # ---- 汇总 ----
        if self.training:
            if self.use_kd and len(kd_loss_list) > 0:
                kd_loss = torch.cat(kd_loss_list, dim=0).mean()
                self.loss_kd['loss_kd'] = self.loss_kd_weight * kd_loss

            if self.use_contrastive and len(contrastive_loss_list) > 0:
                contrastive_loss_temp = torch.stack(contrastive_loss_list).mean()
                self.loss_contrast['loss_contrast'] = self.contrastive_loss_weight * contrastive_loss_temp

            if self.use_kdb and len(kdb_loss_list) > 0:
                kdb_loss = torch.stack(kdb_loss_list).mean()
                self.loss_kdb['loss_kdb'] = self.loss_kdb_weight * kdb_loss


        # -------- BBox 回归 & 分类 --------
        bbox_preds = self.fc_reg(x)

        # 分类余弦相似度
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                                   dim=1).unsqueeze(1).expand_as(self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds

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
        losses = super().loss(cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights, cos_dis,
                              reduction_override)
        if cls_score is not None:
            if self.use_kd:
                losses.update(self.loss_kd)
            if self.use_kdb:
                losses.update(self.loss_kdb)
            if self.use_contrastive:
                losses.update(self.loss_contrast)
        return losses


@HEADS.register_module()
class ArcFaceDisKDBBoxHead(DisKDBBoxHead):
    """DisKDBBoxHead with ArcFaceLoss instead of CrossEntropyLoss."""

    def __init__(self,
                 arcface_loss_cfg=dict(
                     type='ArcFaceLoss',
                     s=64.0,
                     margin=0.5,
                     loss_weight=1.0,
                     reduction='mean'),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_cls = build_loss(arcface_loss_cfg)

    def forward(self, x, return_fc_feat=False):
        return super().forward(x, return_fc_feat)

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
        """ArcFace 替换 CE Loss"""
        losses = dict()
        if cls_score is not None and cls_score.numel() > 0:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                weight=label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)

        # 回归、KD、DisLoss 保持不变
        return super().loss(
            cls_score, bbox_pred, rois, labels,
            label_weights, bbox_targets, bbox_weights,
            cos_dis, reduction_override)
