# Copyright (c) OpenMMLab. All rights reserved.
# Author: ChatGPT (standardized for G-FSOD feature-space visualization)
#
# 功能：
# 1. 支持对任意层特征进行 t-SNE 可视化（默认图像级聚合，更稳）
# 2. 支持直接对分类器权重 fc_cls.weight 做 t-SNE
# 3. 自动区分 base / novel 类别
# 4. 自动适配 split1 / split2
# 5. 输出类别统计信息，便于检查样本分布
#
# 推荐用途：
# - 对比 baseline vs ours 的相对分布趋势
# - 观察 base/novel 的类内紧凑性和类间分离性
# - 观察 classifier weight 的几何组织结构
#
# 不建议用途：
# - 解释全局距离的精确含义
# - 证明“绝对几何正确”
# - 混入大量未筛选 proposal 后直接下结论

import argparse
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmfewshot.detection.datasets import build_dataset, build_dataloader
from mmfewshot.detection.models import build_detector
from mmfewshot.detection.models.neck import FPNWithAdaptiveDCA  # noqa


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Standard t-SNE visualization for G-FSOD features / classifier weights'
    )
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    parser.add_argument(
        '--vis-type',
        type=str,
        default='feature',
        choices=['feature', 'cls_weight'],
        help='visualization target type'
    )
    parser.add_argument(
        '--layer',
        type=str,
        default='',
        help='target layer name for feature mode, e.g. roi_head.bbox_head.novel_shared_fcs.1'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='split1',
        choices=['split1', 'split2'],
        help='dataset split'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='number of images to visualize in feature mode'
    )
    parser.add_argument(
        '--aggregate',
        type=str,
        default='mean',
        choices=['mean'],
        help='feature aggregation strategy for 2D roi features'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='tsne.png',
        help='output image path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )
    parser.add_argument(
        '--perplexity',
        type=float,
        default=-1,
        help='t-SNE perplexity; if < 0, auto choose'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='apply StandardScaler before t-SNE'
    )
    parser.add_argument(
        '--show-class-text',
        action='store_true',
        help='show class id text beside classifier-weight points'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction
    )
    parser.add_argument(
        '--title-fontsize',
        type=int,
        default=16,
        help='font size of figure title'
    )
    parser.add_argument(
        '--text-fontsize',
        type=int,
        default=12,
        help='font size of class text labels'
    )
    parser.add_argument(
        '--hide-title',
        action='store_true',
        help='hide figure title for paper-style multi-panel layout'
    )
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_module_by_name(model, layer_name):
    module = model
    for attr in layer_name.split('.'):
        if not hasattr(module, attr):
            raise ValueError(f"Cannot find submodule '{attr}' in '{layer_name}'")
        module = getattr(module, attr)
    return module


def get_base_novel_class_ids(dataset, split_dict, split='split1'):
    split = split.lower()
    class_to_id = {name: i for i, name in enumerate(dataset.CLASSES)}

    base_names = split_dict[f'BASE_CLASSES_{split.upper()}']
    novel_names = split_dict[f'NOVEL_CLASSES_{split.upper()}']

    for name in list(base_names) + list(novel_names):
        if name not in class_to_id:
            raise KeyError(f"Class '{name}' not found in dataset.CLASSES: {dataset.CLASSES}")

    base_ids = {class_to_id[name] for name in base_names}
    novel_ids = {class_to_id[name] for name in novel_names}
    return base_ids, novel_ids


def flatten_feature_tensor(feat):
    """
    将 hook 到的特征统一展平：
    - 4D: [N, C, H, W] -> GAP -> [N, C]
    - 3D: [N, T, C] or similar -> mean(dim=1) -> [N, C]
    - 2D: keep as [N, C]
    - else: flatten from dim=1
    """
    if feat.ndim == 4:
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
    elif feat.ndim == 3:
        feat = feat.mean(dim=1)
    elif feat.ndim == 2:
        pass
    else:
        feat = feat.view(feat.size(0), -1)
    return feat


def aggregate_feature(feat, mode='mean'):
    """
    对单张图像的特征进行聚合，确保“一个样本一个向量”
    对当前场景，mean 是最稳妥的选择
    """
    if feat.ndim != 2:
        raise ValueError(f'aggregate_feature expects 2D feature, got shape {list(feat.shape)}')

    if mode == 'mean':
        return feat.mean(dim=0, keepdim=True)
    raise ValueError(f'Unsupported aggregate mode: {mode}')


def auto_choose_perplexity(n_samples, user_perplexity=-1):
    if n_samples < 3:
        raise ValueError(f'Not enough samples for t-SNE: {n_samples}')
    if user_perplexity is not None and user_perplexity > 0:
        if user_perplexity >= n_samples:
            return max(1, n_samples - 1)
        return user_perplexity

    # 一个较稳妥的经验式
    perplexity = min(30, max(5, n_samples // 3))
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)
    return perplexity


def build_model_and_dataset(cfg, checkpoint_path):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    model = build_detector(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    return model, dataset, data_loader


def extract_feature_samples(model, dataset, data_loader, layer_name, max_images=100, aggregate='mean'):
    features = {}

    def get_hook(name):
        def hook(module, inputs, output):
            if isinstance(output, (list, tuple)):
                output = output[0]
            if not torch.is_tensor(output):
                print(f"[Warning] Output of {name} is not a tensor, skip.")
                return
            features[name] = output.detach().cpu()
        return hook

    target_module = find_module_by_name(model.module, layer_name)
    hook_handle = target_module.register_forward_hook(get_hook(layer_name))

    feats, labels = [], []
    valid_count = 0

    print(f'[*] Extracting features from layer: {layer_name}')
    print(f'[*] Max images: {max_images}')

    for i, data in enumerate(data_loader):
        if valid_count >= max_images:
            break

        with torch.no_grad():
            _ = model(return_loss=False, rescale=True, **data)

        if layer_name not in features:
            continue

        feat = features.pop(layer_name)
        feat = flatten_feature_tensor(feat)

        # 关键：图像级聚合，避免 proposal 与标签错位
        feat = aggregate_feature(feat, mode=aggregate)

        ann = dataset.get_ann_info(i)
        if len(ann['labels']) == 0:
            continue

        label = int(ann['labels'][0])

        feats.append(feat.numpy())
        labels.append(label)
        valid_count += 1

    hook_handle.remove()

    if len(feats) == 0:
        raise RuntimeError(f'No valid features extracted from layer {layer_name}')

    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels, dtype=np.int64)
    return feats, labels


def extract_classifier_weights(model):
    cls_weight = model.module.roi_head.bbox_head.fc_cls.weight.detach().cpu().numpy()

    # 常见情况：最后一类是背景类
    num_known_classes = len(model.module.CLASSES)
    if cls_weight.shape[0] == num_known_classes + 1:
        cls_weight = cls_weight[:-1]

    labels = np.arange(cls_weight.shape[0], dtype=np.int64)
    return cls_weight, labels


def maybe_normalize_features(feats, do_normalize=False):
    if not do_normalize:
        return feats
    scaler = StandardScaler()
    return scaler.fit_transform(feats)


def run_tsne(feats, perplexity, seed=42):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init='pca',
        learning_rate='auto'
    )
    return tsne.fit_transform(feats)


def print_label_statistics(labels, dataset_classes=None):
    counter = Counter(labels.tolist())
    print('[*] Label distribution:')
    for cls_id in sorted(counter.keys()):
        if cls_id == -1:
            name = 'invalid'
        elif dataset_classes is not None and cls_id < len(dataset_classes):
            name = dataset_classes[cls_id]
        else:
            name = f'class_{cls_id}'
        print(f'    {cls_id:>2}: {name:<20} -> {counter[cls_id]}')


# def plot_tsne_points(
#     X_tsne,
#     labels,
#     base_classes,
#     novel_classes,
#     out_path,
#     title='',
#     show_class_text=False,
#     classifier_weight_mode=False
# ):
#     import colorcet as cc
#     palette = cc.glasbey
#
#     plt.figure(figsize=(10, 8))
#
#     for cls in sorted(set(labels.tolist())):
#         if cls == -1:
#             continue
#
#         idx = labels == cls
#         subset = X_tsne[idx]
#
#         if cls in base_classes:
#             marker = 'o'
#             size = 55 if not classifier_weight_mode else 180
#             edgecolors = 'none'
#             alpha = 0.78
#         elif cls in novel_classes:
#             marker = '^'
#             size = 90 if not classifier_weight_mode else 260
#             edgecolors = 'black'
#             alpha = 0.96
#         else:
#             marker = 's'
#             size = 70 if not classifier_weight_mode else 220
#             edgecolors = 'gray'
#             alpha = 0.85
#
#         plt.scatter(
#             subset[:, 0],
#             subset[:, 1],
#             marker=marker,
#             s=size,
#             color=palette[cls % len(palette)],
#             alpha=alpha,
#             edgecolors=edgecolors,
#             linewidths=0.9
#         )
#
#         if show_class_text or classifier_weight_mode:
#             # classifier weight 模式一般每类一个点
#             for pt in subset:
#                 plt.text(pt[0], pt[1], str(cls), fontsize=10)
#
#     if title:
#         plt.title(title, fontsize=14)
#
#     plt.xticks([])
#     plt.yticks([])
#     plt.tight_layout()
#
#     out_dir = os.path.dirname(out_path)
#     if out_dir:
#         os.makedirs(out_dir, exist_ok=True)
#
#     plt.savefig(out_path, dpi=1200, bbox_inches='tight')
#     plt.close()
#     print(f'[*] Saved t-SNE figure to: {out_path}')

def plot_tsne_points(
    X_tsne,
    labels,
    base_classes,
    novel_classes,
    out_path,
    title='',
    show_class_text=False,
    classifier_weight_mode=False,
    title_fontsize=16,
    text_fontsize=12,
    hide_title=False
):
    import colorcet as cc
    palette = cc.glasbey

    # 单栏四图更适合略偏紧凑的比例
    plt.figure(figsize=(8.0, 6.4), dpi=300)
    ax = plt.gca()
    ax.set_facecolor('white')

    for cls in sorted(set(labels.tolist())):
        if cls == -1:
            continue

        idx = labels == cls
        subset = X_tsne[idx]

        # 更清晰的符号设置
        if cls in base_classes:
            marker = 'o'
            size = 95 if not classifier_weight_mode else 240
            edgecolors = 'black'
            linewidths = 0.9
            alpha = 0.92
        elif cls in novel_classes:
            marker = '^'
            size = 130 if not classifier_weight_mode else 320
            edgecolors = 'black'
            linewidths = 1.1
            alpha = 0.98
        else:
            marker = 's'
            size = 110 if not classifier_weight_mode else 270
            edgecolors = 'black'
            linewidths = 1.0
            alpha = 0.95

        plt.scatter(
            subset[:, 0],
            subset[:, 1],
            marker=marker,
            s=size,
            color=palette[cls % len(palette)],
            alpha=alpha,
            edgecolors=edgecolors,
            linewidths=linewidths
        )

        # 类别编号文本：加粗、更清晰
        if show_class_text or classifier_weight_mode:
            for pt in subset:
                plt.text(
                    pt[0], pt[1], str(cls),
                    fontsize=text_fontsize,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    color='black'
                )

    if title and not hide_title:
        plt.title(title, fontsize=title_fontsize, fontweight='bold', pad=10)

    # 去掉坐标轴刻度，但保留更清晰的边框
    plt.xticks([])
    plt.yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # 给散点稍微留一点呼吸空间
    x_min, x_max = X_tsne[:, 0].min(), X_tsne[:, 0].max()
    y_min, y_max = X_tsne[:, 1].min(), X_tsne[:, 1].max()
    x_pad = (x_max - x_min) * 0.06 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.06 if y_max > y_min else 1.0
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_path, dpi=1200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[*] Saved t-SNE figure to: {out_path}')


def main():
    args = parse_args()
    set_random_seed(args.seed)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model, dataset, data_loader = build_model_and_dataset(cfg, args.checkpoint)
    base_classes, novel_classes = get_base_novel_class_ids(dataset, NWPUV2_SPLIT, args.split)

    print('[*] dataset.CLASSES =', dataset.CLASSES)
    print('[*] model.CLASSES   =', model.module.CLASSES)
    print('[*] base class ids  =', sorted(base_classes))
    print('[*] novel class ids =', sorted(novel_classes))

    if args.vis_type == 'feature':
        if not args.layer:
            raise ValueError('--layer must be provided when --vis-type feature')

        feats, labels = extract_feature_samples(
            model=model,
            dataset=dataset,
            data_loader=data_loader,
            layer_name=args.layer,
            max_images=args.samples,
            aggregate=args.aggregate
        )

        print(f'[*] Extracted feature shape: {feats.shape}')
        print_label_statistics(labels, dataset_classes=dataset.CLASSES)

        feats = maybe_normalize_features(feats, do_normalize=args.normalize)
        perplexity = auto_choose_perplexity(len(feats), args.perplexity)
        print(f'[*] Using perplexity={perplexity}')

        X_tsne = run_tsne(feats, perplexity=perplexity, seed=args.seed)

        title = f't-SNE of {args.layer}'
        plot_tsne_points(
            X_tsne=X_tsne,
            labels=labels,
            base_classes=base_classes,
            novel_classes=novel_classes,
            out_path=args.out,
            title=title,
            show_class_text=args.show_class_text,
            classifier_weight_mode=False,
            title_fontsize=args.title_fontsize,
            text_fontsize=args.text_fontsize,
            hide_title=args.hide_title
        )

    elif args.vis_type == 'cls_weight':
        feats, labels = extract_classifier_weights(model)

        print(f'[*] Extracted classifier weight shape: {feats.shape}')
        print_label_statistics(labels, dataset_classes=dataset.CLASSES)

        feats = maybe_normalize_features(feats, do_normalize=args.normalize)
        perplexity = auto_choose_perplexity(len(feats), args.perplexity)
        print(f'[*] Using perplexity={perplexity}')

        X_tsne = run_tsne(feats, perplexity=perplexity, seed=args.seed)

        title = 't-SNE of fc_cls.weight'
        plot_tsne_points(
            X_tsne=X_tsne,
            labels=labels,
            base_classes=base_classes,
            novel_classes=novel_classes,
            out_path=args.out,
            title=title,
            show_class_text=True,
            classifier_weight_mode=True,
            title_fontsize=args.title_fontsize,
            text_fontsize=args.text_fontsize,
            hide_title=args.hide_title
        )

    else:
        raise ValueError(f'Unsupported vis-type: {args.vis_type}')


if __name__ == '__main__':
    main()