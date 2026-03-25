# # Copyright (c) OpenMMLab. All rights reserved.
# # Author: ChatGPT (Rock Jim customized)
# #
# # 功能：在MMDetection/MMFewShot框架中对FPN的C3层特征进行t-SNE可视化，
# #       base类和novel类使用不同符号，不同类别使用不同颜色。
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


NWPUV2_SPLIT = dict(
    ALL_CLASSES_SPLIT1=('airplane', 'baseball', 'basketball', 'bridge',
                        'groundtrackfield', 'harbor', 'ship',
                        'storagetank', 'tenniscourt', 'vehicle'),
    NOVEL_CLASSES_SPLIT1=('airplane', 'baseball', 'tenniscourt'),
    BASE_CLASSES_SPLIT1=('basketball', 'bridge', 'groundtrackfield',
                         'harbor', 'ship', 'storagetank', 'vehicle'),

    ALL_CLASSES_SPLIT2=('airplane', 'baseball', 'basketball', 'bridge',
                        'groundtrackfield', 'harbor', 'ship',
                        'storagetank', 'tenniscourt', 'vehicle'),
    NOVEL_CLASSES_SPLIT2=('basketball', 'groundtrackfield', 'vehicle'),
    BASE_CLASSES_SPLIT2=('airplane', 'baseball', 'bridge',
                         'harbor', 'ship', 'storagetank', 'tenniscourt'),
)


def get_base_novel_class_ids(dataset, split_dict, split='split1'):
    split = split.lower()
    if split not in ['split1', 'split2']:
        raise ValueError(f'Unsupported split: {split}')

    class_to_id = {name: i for i, name in enumerate(dataset.CLASSES)}

    base_names = split_dict[f'BASE_CLASSES_{split.upper()}']
    novel_names = split_dict[f'NOVEL_CLASSES_{split.upper()}']

    # 检查名称是否都在 dataset.CLASSES 中
    for name in list(base_names) + list(novel_names):
        if name not in class_to_id:
            raise KeyError(
                f"Class '{name}' not found in dataset.CLASSES: {dataset.CLASSES}"
            )

    base_ids = {class_to_id[name] for name in base_names}
    novel_ids = {class_to_id[name] for name in novel_names}

    return base_ids, novel_ids


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
        # --- 控制每张图的 RoI 数量 ---
        if feat.ndim == 2:
            # feat shape = [num_roi, feat_dim]

            max_rois = 200  # 每图最多三个 RoI

            if feat.size(0) > max_rois:
                idx = np.random.choice(feat.size(0), max_rois, replace=False)
                feat = feat[idx]


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
    # base_classes = set(range(0, 6))
    # novel_classes = set(range(7, 9))
    base_classes, novel_classes = get_base_novel_class_ids(
        dataset, NWPUV2_SPLIT, split='split1'
    )

    plt.figure(figsize=(10, 8))

    # palette = sns.color_palette('hls', len(set(labels)))
    # palette = sns.color_palette('tab20', len(set(labels)))
    import colorcet as cc
    palette = cc.glasbey  # 高区分度色板

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

    plt.title(f't-SNE Visualization of {args.layer} Features', fontsize=20)
    plt.legend(ncol=1, bbox_to_anchor=(1, 1), loc='upper left', fontsize=30)
    plt.tight_layout()
    plt.savefig(args.out, dpi=1200)
    print(f'[*] Saved visualization to {args.out}')


if __name__ == '__main__':
    main()
