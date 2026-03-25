# # Copyright (c) OpenMMLab. All rights reserved.
# # Author: ChatGPT (Rock Jim customized)
# #
# # 功能：在MMDetection/MMFewShot框架中对FPN的C3层特征进行t-SNE可视化，
# #       base类和novel类使用不同符号，不同类别使用不同颜色。
# #python tools/analysis_tools/tsne_feature_visualization.py \
#     # configs/tfa_r101_fpn_voc_split1_base.py \
#     # work_dirs/tfa_base/latest_arcfaceloss.pth \
#     # --layer C3 \
#     # --samples 200 \
#     # --out tsne_fpn_c3.png
#
#
# import argparse
# import os
# import warnings
# import random
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch.nn.functional as F
#
# from sklearn.manifold import TSNE
# from mmcv import Config, DictAction
# from mmcv.runner import load_checkpoint
# from mmcv.parallel import MMDataParallel
# from mmfewshot.detection.datasets import build_dataset, build_dataloader
# from mmfewshot.detection.models import build_detector
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='t-SNE visualization of FPN features')
#     parser.add_argument('config', help='test config file path')
#     parser.add_argument('checkpoint', help='checkpoint file')
#     parser.add_argument('--samples', type=int, default=200,
#                         help='number of images to visualize (default: 200)')
#     parser.add_argument('--layer', type=str, default='C3',
#                         help='target FPN layer: C3/C4/C5 (default: C3)')
#     parser.add_argument('--out', type=str, default='tsne_fpn.png',
#                         help='path to save the output visualization')
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--cfg-options', nargs='+', action=DictAction)
#     args = parser.parse_args()
#     return args
#
#
# def main():
#     args = parse_args()
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#
#     cfg = Config.fromfile(args.config)
#     if args.cfg_options is not None:
#         cfg.merge_from_dict(args.cfg_options)
#
#     cfg.model.pretrained = None
#     cfg.model.train_cfg = None
#
#     dataset = build_dataset(cfg.data.test)
#     data_loader = build_dataloader(
#         dataset,
#         samples_per_gpu=1,
#         workers_per_gpu=cfg.data.workers_per_gpu,
#         dist=False,
#         shuffle=False)
#
#     # build model
#     model = build_detector(cfg.model)
#     checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
#     model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
#     model = MMDataParallel(model, device_ids=[0])
#     model.eval()
#
#     # ------------------------
#     # 注册 FPN 层 Hook
#     # ------------------------
#     features = {}
#
#     def get_hook(name):
#         def hook(model, input, output):
#             output = output.clone().detach().cpu()
#             if name not in features:
#                 features[name] = []
#             features[name].append(output)
#             print(f"[Hook] Captured {name} feature with shape {output.shape}")
#
#         return hook
#
#     # FPN 的 lateral_convs 对应 C3, C4, C5, C6
#     fpn_layer_map = {'C3': 0, 'C4': 1, 'C5': 2, 'C6': 3}
#     target_idx = fpn_layer_map[args.layer]
#     model.module.neck.lateral_convs[target_idx].register_forward_hook(get_hook(args.layer))
#
#     # ------------------------
#     # 前向推理收集特征
#     # ------------------------
#     feats, labels = [], []
#     print(f'[*] Extracting {args.layer} features from {args.samples} samples...')
#
#     for i, data in enumerate(data_loader):
#         if i >= args.samples:
#             break
#         with torch.no_grad():
#             _ = model(return_loss=False, rescale=True, **data)
#
#         # ✅ 获取并清空当前批次特征
#         feat_list = features.pop(args.layer)
#         feat = torch.cat(feat_list, dim=0)
#
#         # ✅ 全局池化
#         feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
#         feats.append(feat.numpy())
#
#         # ✅ 提取标签
#         if hasattr(dataset, 'get_ann_info'):
#             ann = dataset.get_ann_info(i)
#             if len(ann['labels']) > 0:
#                 labels.append(int(ann['labels'][0]))
#             else:
#                 labels.append(-1)
#         else:
#             labels.append(-1)
#
#     feats = np.concatenate(feats, axis=0)
#     labels = np.array(labels)
#
#     # ------------------------
#     # t-SNE 降维可视化
#     # ------------------------
#     print('[*] Running t-SNE...')
#     tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed)
#     X_tsne = tsne.fit_transform(feats)
#
#     # base 与 novel 区分 (根据类别编号划分)
#     base_classes = set(range(0, 15))
#     novel_classes = set(range(15, 20))
#
#     plt.figure(figsize=(10, 8))
#     palette = sns.color_palette('hls', len(set(labels)))
#
#     for cls in sorted(set(labels)):
#         if cls == -1:
#             continue
#         idx = labels == cls
#         subset = X_tsne[idx]
#         if cls in base_classes:
#             marker = 'o'
#         elif cls in novel_classes:
#             marker = 'x'
#         else:
#             marker = '+'
#         plt.scatter(subset[:, 0], subset[:, 1],
#                     label=f'Class {cls}',
#                     marker=marker,
#                     color=palette[cls % len(palette)],
#                     alpha=0.7,
#                     s=35)
#
#     plt.title(f't-SNE Visualization of FPN {args.layer} Features')
#     plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
#     plt.tight_layout()
#     plt.savefig(args.out, dpi=300)
#     print(f'[*] Saved visualization to {args.out}')
#
#
# if __name__ == '__main__':
#     main()

# Copyright (c) OpenMMLab. All rights reserved.
# Author: ChatGPT (Rock Jim customized)
#
# 功能：可从 MMDetection/MMFewShot 模型任意层提取特征并进行 t-SNE 可视化。
# 支持 FPN 层、ROI Head 层、分类器层等任意位置。
#
# 使用示例：
# python tools/analysis_tools/tsne_feature_visualization.py \
#   configs/detection/ETF/dior/split1/ETF_r101_fpn_dior-split1_10shot-fine-tuning.py \
#   work_dirs/dior/rep/tfa_r101_fpn_dior_split1_base-training/ETF_r101_fpn_dior-split1_10shot-fine-tuning/latest_arcfaceloss.pth \
#   --layer roi_head.bbox_head.shared_fcs.1 \
#   --samples 200 \
#   --out tsne_shared_fcs.png

import argparse
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.manifold import TSNE
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmfewshot.detection.datasets import build_dataset, build_dataloader
from mmfewshot.detection.models import build_detector
from mmfewshot.detection.models.neck import FPNWithAdaptiveDCA # noqa


def parse_args():
    parser = argparse.ArgumentParser(description='t-SNE visualization of arbitrary model features')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', type=int, default=200, help='number of images to visualize')
    parser.add_argument('--layer', type=str, required=True, help='target layer name (e.g., backbone.layer4, roi_head.bbox_head.shared_fcs.1)')
    parser.add_argument('--out', type=str, default='tsne_features.png', help='path to save output visualization')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    return args


def find_module_by_name(model, layer_name):
    """递归查找模型中指定名称的模块"""
    module = model
    for attr in layer_name.split('.'):
        if not hasattr(module, attr):
            raise ValueError(f"[Error] Cannot find submodule '{attr}' in '{layer_name}'")
        module = getattr(module, attr)
    return module


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # ------------------------
    # 构建模型并加载权重
    # ------------------------
    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # ------------------------
    # 注册 Hook
    # ------------------------
    features = {}

    def get_hook(name):
        def hook(model, input, output):
            output = output.clone().detach().cpu()
            features[name] = output
            print(f"[Hook] Captured {name} with shape {list(output.shape)}")
        return hook

    target_module = find_module_by_name(model.module, args.layer)
    target_module.register_forward_hook(get_hook(args.layer))

    # ------------------------
    # 提取特征
    # ------------------------
    feats, labels = [], []
    print(f'[*] Extracting {args.layer} features from {args.samples} samples...')

    for i, data in enumerate(data_loader):
        if i >= args.samples:
            break
        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)

        if args.layer not in features:
            continue

        feat = features.pop(args.layer)

        # ✅ 通用特征展平逻辑
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)
        elif feat.ndim == 2:
            feat = feat
        else:
            feat = feat.view(feat.size(0), -1)

        feats.append(feat.numpy())

        # if hasattr(dataset, 'get_ann_info'):
        #     ann = dataset.get_ann_info(i)
        #     if len(ann['labels']) > 0:
        #         labels.append(int(ann['labels'][0]))
        #     else:
        #         labels.append(-1)
        # else:
        #     labels.append(-1)

        # 提取标签（对应每个 RoI）
        if hasattr(dataset, 'get_ann_info'):
            ann = dataset.get_ann_info(i)
            roi_labels = []
            if len(ann['labels']) > 0:
                for _ in range(feat.shape[0]):  # 每个 RoI
                    roi_labels.append(int(ann['labels'][0]))  # 简单策略：所有 RoI 取第一目标类别
            else:
                roi_labels = [-1] * feat.shape[0]
        else:
            roi_labels = [-1] * feat.shape[0]

        labels.extend(roi_labels)

    if len(feats) == 0:
        raise RuntimeError(f"No features captured from layer {args.layer}")

    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)

    # ------------------------
    # t-SNE 可视化
    # ------------------------
    print('[*] Running t-SNE...')
    tsne = TSNE(n_components=2, perplexity=30, random_state=args.seed)
    X_tsne = tsne.fit_transform(feats)

    # base 与 novel 类划分
    base_classes = set(range(0, 15))
    novel_classes = set(range(15, 20))

    plt.figure(figsize=(10, 8))
    # palette = sns.color_palette('hls', len(set(labels)))
    palette = sns.color_palette('tab20', len(set(labels)))

    for cls in sorted(set(labels)):
        if cls == -1:
            continue
        idx = labels == cls
        subset = X_tsne[idx]
        if cls in base_classes:
            marker = 'o'
        elif cls in novel_classes:
            marker = 'x'
        else:
            marker = '+'
        plt.scatter(subset[:, 0], subset[:, 1],
                    label=f'Class {cls}',
                    marker=marker,
                    color=palette[cls % len(palette)],
                    alpha=0.7,
                    s=35)

    plt.title(f't-SNE Visualization of {args.layer} Features')
    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f'[*] Saved visualization to {args.out}')


if __name__ == '__main__':
    main()
