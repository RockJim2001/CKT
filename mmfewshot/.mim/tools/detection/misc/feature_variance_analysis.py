# Copyright (c) OpenMMLab. All rights reserved.
# Author: ChatGPT (Rock Jim customized)
#
# 功能：
#   从指定层提取特征，计算类内方差与类间距离（intra-class variance / inter-class distance）。
#   可用于比较不同模块（如 FPN 各层、ROI Head、fc_cls 等）的判别能力。
#
# 用法示例：
# python tools/analysis_tools/feature_variance_analysis.py \
#   configs/detection/ETF/dior/split1/ETF_r101_fpn_dior-split1_10shot-fine-tuning.py \
#   work_dirs/dior/.../latest_arcfaceloss.pth \
#   --layer roi_head.bbox_head.fc_cls \
#   --samples 200 \
#   --out results_fc_cls.txt

import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F

from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmfewshot.detection.datasets import build_dataset, build_dataloader
from mmfewshot.detection.models import build_detector
from mmfewshot.detection.models.neck import FPNWithAdaptiveDCA  # noqa


# ------------------------------------------------------------
# 工具函数：递归查找模块
# ------------------------------------------------------------
def find_module_by_name(model, layer_name):
    module = model
    for attr in layer_name.split('.'):
        if not hasattr(module, attr):
            raise ValueError(f"[Error] Cannot find submodule '{attr}' in '{layer_name}'")
        module = getattr(module, attr)
    return module


# ------------------------------------------------------------
# 计算类内方差 & 类间距离
# ------------------------------------------------------------
def compute_intra_inter_distance(features, labels):
    features = np.array(features)
    labels = np.array(labels)
    unique_labels = np.unique(labels[labels >= 0])

    class_centers = []
    intra_var = 0.0

    for c in unique_labels:
        class_feats = features[labels == c]
        mu_c = class_feats.mean(axis=0)
        class_centers.append(mu_c)
        intra_var += np.mean(np.linalg.norm(class_feats - mu_c, axis=1) ** 2)

    intra_var /= len(unique_labels)
    class_centers = np.stack(class_centers, axis=0)

    # 类间距离
    from scipy.spatial.distance import cdist
    inter_dists = cdist(class_centers, class_centers)
    inter_dists = inter_dists[np.triu_indices(len(unique_labels), k=1)]
    inter_mean = inter_dists.mean()

    sep_ratio = inter_mean / (intra_var + 1e-6)
    return intra_var, inter_mean, sep_ratio


# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Feature variance analysis for arbitrary layer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--out', type=str, default='feature_variance.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


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

        # 展平维度
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)
        elif feat.ndim == 2:
            feat = feat
        else:
            feat = feat.view(feat.size(0), -1)

        feats.append(feat.numpy())

        # === 🔧 修复部分：为每个目标分配标签 ===
        if hasattr(dataset, 'get_ann_info'):
            ann = dataset.get_ann_info(i)
            if len(ann['labels']) > 0:
                # 若目标数与特征数不同，可进行截断或填充
                num_obj = feat.shape[0]
                labels_per_img = np.array(ann['labels'])
                # 如果目标数量少于特征数，用最后一个标签重复补齐
                if len(labels_per_img) < num_obj:
                    labels_per_img = np.pad(
                        labels_per_img,
                        (0, num_obj - len(labels_per_img)),
                        mode='edge'
                    )
                else:
                    labels_per_img = labels_per_img[:num_obj]
                labels.extend(labels_per_img.tolist())
            else:
                labels.extend([-1] * feat.shape[0])
        else:
            labels.extend([-1] * feat.shape[0])

    if len(feats) == 0:
        raise RuntimeError(f"No features captured from layer {args.layer}")

    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)

    # ------------------------
    # 计算类内/类间距离
    # ------------------------
    print('[*] Computing intra/inter class statistics...')
    intra_var, inter_dist, sep_ratio = compute_intra_inter_distance(feats, labels)

    # 输出结果
    with open(args.out, 'w') as f:
        f.write(f"Layer: {args.layer}\n")
        f.write(f"Samples: {args.samples}\n")
        f.write(f"Intra-class variance: {intra_var:.6f}\n")
        f.write(f"Inter-class distance: {inter_dist:.6f}\n")
        f.write(f"Separation ratio (Inter / Intra): {sep_ratio:.6f}\n")

    print(f"[+] Intra-class variance: {intra_var:.6f}")
    print(f"[+] Inter-class distance: {inter_dist:.6f}")
    print(f"[+] Separation ratio (Inter/Intra): {sep_ratio:.6f}")
    print(f"[*] Results saved to {args.out}")


if __name__ == '__main__':
    main()
