import argparse
import os
import numpy as np
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from mmfewshot.detection.core import eval_map


NWPUV2_SPLIT = dict(
    ALL_CLASSES_SPLIT1=('airplane', 'baseball', 'basketball', 'bridge',
                        'groundtrackfield', 'harbor', 'ship', 'storagetank',
                        'tenniscourt', 'vehicle'),
    NOVEL_CLASSES_SPLIT1=('airplane', 'baseball', 'tenniscourt'),
    BASE_CLASSES_SPLIT1=('basketball', 'bridge', 'groundtrackfield',
                         'harbor', 'ship', 'storagetank', 'vehicle'),

    ALL_CLASSES_SPLIT2=('airplane', 'baseball', 'basketball', 'bridge',
                        'groundtrackfield', 'harbor', 'ship', 'storagetank',
                        'tenniscourt', 'vehicle'),
    NOVEL_CLASSES_SPLIT2=('basketball', 'groundtrackfield', 'vehicle'),
    BASE_CLASSES_SPLIT2=('airplane', 'baseball', 'bridge',
                         'harbor', 'ship', 'storagetank', 'tenniscourt'),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate percentile-based scale-wise Novel AP on NWPU VHR-10.v2 from result.pkl')
    parser.add_argument(
        '--config',
        required=True,
        help='Path to config file')
    parser.add_argument(
        '--result-pkl',
        required=True,
        help='Path to result.pkl')
    parser.add_argument(
        '--split',
        default='split1',
        choices=['split1', 'split2'],
        help='Dataset split')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for mAP')
    parser.add_argument(
        '--q1',
        type=float,
        default=1.0 / 3.0,
        help='First percentile threshold, default=1/3')
    parser.add_argument(
        '--q2',
        type=float,
        default=2.0 / 3.0,
        help='Second percentile threshold, default=2/3')
    return parser.parse_args()


def bbox_areas(bboxes):
    if len(bboxes) == 0:
        return np.array([], dtype=np.float32)
    wh = np.maximum(bboxes[:, 2:] - bboxes[:, :2], 0)
    return wh[:, 0] * wh[:, 1]


def get_novel_classes(split='split1'):
    key = f'NOVEL_CLASSES_{split.upper()}'
    return NWPUV2_SPLIT[key]


def filter_results_by_class(results, valid_label_ids, num_classes):
    filtered_results = []
    for img_res in results:
        new_img_res = []
        for cls_id in range(num_classes):
            if cls_id in valid_label_ids:
                new_img_res.append(img_res[cls_id])
            else:
                new_img_res.append(np.zeros((0, 5), dtype=np.float32))
        filtered_results.append(new_img_res)
    return filtered_results


def get_img_wh(dataset, idx):
    """
    尽量兼容 NWPUVHR10Dataset 的 data_infos / img_info 写法
    """
    if hasattr(dataset, 'data_infos'):
        info = dataset.data_infos[idx]
        if 'width' in info and 'height' in info:
            return float(info['width']), float(info['height'])
        if 'img_info' in info:
            img_info = info['img_info']
            if 'width' in img_info and 'height' in img_info:
                return float(img_info['width']), float(img_info['height'])

    # 兜底：从 ann_info 里拿不到图像尺寸时，报错提醒
    raise ValueError(
        f'Cannot find image width/height for dataset index {idx}. '
        f'Please check dataset.data_infos format.')


def collect_novel_relative_areas(dataset, novel_label_ids):
    """
    收集所有 Novel GT 的相对面积比例:
        ratio = bbox_area / image_area
    """
    ratios = []
    for idx in range(len(dataset)):
        ann = dataset.get_ann_info(idx)
        bboxes = ann['bboxes']
        labels = ann['labels']

        if len(bboxes) == 0:
            continue

        img_w, img_h = get_img_wh(dataset, idx)
        img_area = img_w * img_h
        areas = bbox_areas(bboxes)

        keep = np.isin(labels, novel_label_ids)
        selected_areas = areas[keep]

        if len(selected_areas) > 0:
            ratios.extend((selected_areas / img_area).tolist())

    return np.array(ratios, dtype=np.float32)


def get_ratio_ranges_from_quantiles(ratios, q1=1/3, q2=2/3):
    assert len(ratios) > 0, 'No Novel GT ratios found.'
    t1 = float(np.quantile(ratios, q1))
    t2 = float(np.quantile(ratios, q2))

    return {
        'AP_S': (0.0, t1),
        'AP_M': (t1, t2),
        'AP_L': (t2, 1.0),
    }, t1, t2


def filter_ann_by_class_and_ratio(dataset, idx, valid_label_ids, min_ratio, max_ratio, is_last_bucket=False):
    ann = dataset.get_ann_info(idx)
    bboxes = ann['bboxes']
    labels = ann['labels']

    if len(bboxes) == 0:
        return dict(
            bboxes=np.zeros((0, 4), dtype=np.float32),
            labels=np.array([], dtype=np.int64),
            bboxes_ignore=np.zeros((0, 4), dtype=np.float32),
            labels_ignore=np.array([], dtype=np.int64))

    img_w, img_h = get_img_wh(dataset, idx)
    img_area = img_w * img_h
    ratios = bbox_areas(bboxes) / img_area

    if is_last_bucket:
        keep = np.isin(labels, valid_label_ids) & (ratios >= min_ratio) & (ratios <= max_ratio)
    else:
        keep = np.isin(labels, valid_label_ids) & (ratios >= min_ratio) & (ratios < max_ratio)

    return dict(
        bboxes=bboxes[keep],
        labels=labels[keep],
        bboxes_ignore=np.zeros((0, 4), dtype=np.float32),
        labels_ignore=np.array([], dtype=np.int64))


def count_gt_info(filtered_annotations):
    total_gts = 0
    class_counter = {}
    for ann in filtered_annotations:
        labels = ann['labels']
        total_gts += len(labels)
        for lb in labels.tolist():
            class_counter[lb] = class_counter.get(lb, 0) + 1
    return total_gts, class_counter


def main():
    args = parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f'Config not found: {args.config}')
    if not os.path.isfile(args.result_pkl):
        raise FileNotFoundError(f'result.pkl not found: {args.result_pkl}')

    if not (0.0 < args.q1 < args.q2 < 1.0):
        raise ValueError(f'Invalid quantiles: q1={args.q1}, q2={args.q2}')

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result_pkl)

    if len(results) != len(dataset):
        raise ValueError(
            f'Length mismatch: len(results)={len(results)} vs len(dataset)={len(dataset)}')

    class_names = dataset.CLASSES
    novel_class_names = get_novel_classes(args.split)
    novel_label_ids = [class_names.index(c) for c in novel_class_names]

    print('=' * 80)
    print(f'All classes: {class_names}')
    print(f'Novel classes ({args.split}): {novel_class_names}')
    print(f'Novel label ids: {novel_label_ids}')
    print('=' * 80)

    # 1. 收集 Novel GT 的相对面积比例
    novel_ratios = collect_novel_relative_areas(dataset, novel_label_ids)
    print(f'Total Novel GT ratios collected: {len(novel_ratios)}')

    # 2. 基于百分位数得到三个尺度桶
    ratio_ranges, t1, t2 = get_ratio_ranges_from_quantiles(
        novel_ratios, q1=args.q1, q2=args.q2)

    print(f'Quantile thresholds:')
    print(f'  q1={args.q1:.4f} -> t1={t1:.8f}')
    print(f'  q2={args.q2:.4f} -> t2={t2:.8f}')
    print('Scale bins based on relative area ratio (bbox_area / image_area):')
    for k, (l, r) in ratio_ranges.items():
        print(f'  {k}: [{l:.8f}, {r:.8f}]')
    print('=' * 80)

    # 3. 只保留 Novel 类预测
    filtered_results = filter_results_by_class(results, novel_label_ids, len(class_names))

    # 4. 分桶评估
    final_results = {}

    bucket_names = ['AP_S', 'AP_M', 'AP_L']
    for i, metric_name in enumerate(bucket_names):
        min_ratio, max_ratio = ratio_ranges[metric_name]
        is_last_bucket = (metric_name == 'AP_L')

        filtered_annotations = []
        for idx in range(len(dataset)):
            filtered_ann = filter_ann_by_class_and_ratio(
                dataset,
                idx,
                novel_label_ids,
                min_ratio,
                max_ratio,
                is_last_bucket=is_last_bucket)
            filtered_annotations.append(filtered_ann)

        total_gts, class_counter = count_gt_info(filtered_annotations)
        valid_cls_num = len(class_counter)

        print(f'\n[{metric_name}] ratio in [{min_ratio:.8f}, {max_ratio:.8f}]')
        print(f'  Total Novel GTs: {total_gts}')
        print(f'  Covered Novel classes: {valid_cls_num}/{len(novel_label_ids)}')

        if total_gts == 0:
            final_results[metric_name] = np.nan
            print(f'  {metric_name}: NaN (no valid GTs)')
            continue

        mean_ap, _ = eval_map(
            filtered_results,
            filtered_annotations,
            class_names,
            scale_ranges=None,
            iou_thr=args.iou_thr,
            logger='silent'
        )
        final_results[metric_name] = mean_ap
        print(f'  {metric_name}: {mean_ap:.4f}')

    print('\n' + '=' * 80)
    print('Percentile-based scale-wise Novel AP results:')
    for k, v in final_results.items():
        if np.isnan(v):
            print(f'{k}: NaN')
        else:
            print(f'{k}: {v:.4f}')
    print('=' * 80)


if __name__ == '__main__':
    main()


# baseline
# python tools/detection/analysis_tools/eval_nwpuv2_novel_scale_ap.py --config configs/detection/GFSDet/nwpuv2/split1-ori/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_nwpu-split1_10shot-fine-tuning.py --result-pkl work_dirs/nwpuv2_rep/split1-ori/AdaptiveResidual/tfa_r101_fpn_nwpuv2_split1_base-training/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_nwpu-split1_10shot-fine-tuning/result.pkl --split split1

# baseline + AFPN
# python tools/detection/analysis_tools/eval_nwpuv2_novel_scale_ap.py --config configs/detection/GFSDet/nwpuv2/split1-ablation/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_nwpu-split1_10shot-fine-tuning.py --result-pkl work_dirs/nwpuv2_rep/split1/AdaptiveResidual/tfa_r101_fpn_nwpuv2_split1_base-training/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_nwpu-split1_10shot-fine-tuning/NovelSharedKnowledgeTransferFusion/AFPN-ONLY/result.pkl --split split1

# full model
# python tools/detection/analysis_tools/eval_nwpuv2_novel_scale_ap.py --config configs/detection/GFSDet/nwpuv2/split1/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_nwpu-split1_10shot-fine-tuning.py --result-pkl work_dirs/nwpuv2_rep/split1/AdaptiveResidual/tfa_r101_fpn_nwpuv2_split1_base-training/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_nwpu-split1_10shot-fine-tuning/NovelSharedKnowledgeTransferFusion/result.pkl --split split1