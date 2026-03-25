import argparse
import os

import mmcv
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MultipleLocator

from mmcv import Config, DictAction
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmfewshot.detection.datasets import build_dataset
from mmfewshot.detection.models.neck import FPNWithAdaptiveDCA  # noqa


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix for mmfewshot/mmdet2.x results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('prediction_path', help='path of test results pkl')
    parser.add_argument('save_dir', help='directory to save confusion matrix')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection boxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for TP matching')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to show the figure')
    parser.add_argument(
        '--color-theme',
        default='Blues',
        help='colormap theme, e.g., Blues, viridis')
    parser.add_argument(
        '--with-background',
        action='store_true',
        help='whether to include background row/column')
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='normalize each row to percentage')
    parser.add_argument(
        '--hide-title',
        action='store_true',
        help='hide figure title for paper-style plots')
    parser.add_argument(
        '--split',
        type=str,
        default='split1',
        choices=['split1', 'split2'],
        help='dataset split')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in config')
    args = parser.parse_args()
    return args


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(cfg):
    test_cfg = cfg.data.test
    dataset = build_dataset(test_cfg)
    return dataset


def get_classes(dataset):
    if hasattr(dataset, 'CLASSES') and dataset.CLASSES is not None:
        return dataset.CLASSES
    raise ValueError('Cannot find dataset.CLASSES. Please check dataset setup.')


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

    for name in list(base_names) + list(novel_names):
        if name not in class_to_id:
            raise KeyError(
                f"Class '{name}' not found in dataset.CLASSES: {dataset.CLASSES}"
            )

    base_ids = {class_to_id[name] for name in base_names}
    novel_ids = {class_to_id[name] for name in novel_names}
    return base_ids, novel_ids


def filter_dets(per_img_res, score_thr=0.3):
    if isinstance(per_img_res, tuple):
        per_img_res = per_img_res[0]

    det_bboxes = []
    det_scores = []
    det_labels = []

    for cls_id, cls_res in enumerate(per_img_res):
        if cls_res is None or len(cls_res) == 0:
            continue

        cls_res = np.asarray(cls_res)
        if cls_res.ndim != 2 or cls_res.shape[1] < 5:
            continue

        scores = cls_res[:, 4]
        keep = scores >= score_thr
        cls_res = cls_res[keep]

        if len(cls_res) == 0:
            continue

        det_bboxes.append(cls_res[:, :4])
        det_scores.append(cls_res[:, 4])
        det_labels.append(np.full((len(cls_res),), cls_id, dtype=np.int64))

    if len(det_bboxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64)
        )

    det_bboxes = np.concatenate(det_bboxes, axis=0).astype(np.float32)
    det_scores = np.concatenate(det_scores, axis=0).astype(np.float32)
    det_labels = np.concatenate(det_labels, axis=0).astype(np.int64)

    order = np.argsort(-det_scores)
    det_bboxes = det_bboxes[order]
    det_scores = det_scores[order]
    det_labels = det_labels[order]

    return det_bboxes, det_scores, det_labels


def match_dets_to_gts(det_bboxes, det_labels, gt_bboxes, gt_labels, iou_thr=0.5):
    num_gt = len(gt_bboxes)
    num_det = len(det_bboxes)

    if num_gt == 0 and num_det == 0:
        return [], [], []
    if num_gt == 0:
        return [], [], list(range(num_det))
    if num_det == 0:
        return [], list(range(num_gt)), []

    det_bboxes_t = torch.from_numpy(det_bboxes).float()
    gt_bboxes_t = torch.from_numpy(gt_bboxes).float()

    ious = bbox_overlaps(det_bboxes_t, gt_bboxes_t).cpu().numpy()

    matched_gt = set()
    matched_det = set()
    matches = []

    for det_idx in range(num_det):
        best_gt = -1
        best_iou = -1.0

        for gt_idx in range(num_gt):
            if gt_idx in matched_gt:
                continue
            iou = ious[det_idx, gt_idx]
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_gt >= 0:
            matched_gt.add(best_gt)
            matched_det.add(det_idx)
            matches.append((best_gt, det_idx))

    unmatched_gt = [i for i in range(num_gt) if i not in matched_gt]
    unmatched_det = [i for i in range(num_det) if i not in matched_det]

    return matches, unmatched_gt, unmatched_det


def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0.3,
                               tp_iou_thr=0.5,
                               with_background=True):
    classes = get_classes(dataset)
    num_classes = len(classes)

    if with_background:
        cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.float64)
        bg_idx = num_classes
    else:
        cm = np.zeros((num_classes, num_classes), dtype=np.float64)
        bg_idx = None

    assert len(dataset) == len(results), \
        'Dataset length and result length must be the same.'

    prog_bar = mmcv.ProgressBar(len(results))

    for idx, per_img_res in enumerate(results):
        ann = dataset.get_ann_info(idx)
        gt_bboxes = np.array(ann['bboxes'], dtype=np.float32).reshape(-1, 4)
        gt_labels = np.array(ann['labels'], dtype=np.int64).reshape(-1)

        det_bboxes, det_scores, det_labels = filter_dets(
            per_img_res, score_thr=score_thr)

        matches, unmatched_gt, unmatched_det = match_dets_to_gts(
            det_bboxes, det_labels, gt_bboxes, gt_labels, iou_thr=tp_iou_thr)

        for gt_idx, det_idx in matches:
            gt_label = int(gt_labels[gt_idx])
            pred_label = int(det_labels[det_idx])
            cm[gt_label, pred_label] += 1

        if with_background:
            for gt_idx in unmatched_gt:
                gt_label = int(gt_labels[gt_idx])
                cm[gt_label, bg_idx] += 1

            for det_idx in unmatched_det:
                pred_label = int(det_labels[det_idx])
                cm[bg_idx, pred_label] += 1

        prog_bar.update()

    return cm


def normalize_confusion_matrix(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm.astype(np.float32) / row_sums * 100.0


def draw_symbolic_ticklabels(ax,
                             labels,
                             base_classes,
                             novel_classes,
                             with_background,
                             palette,
                             tick_fontsize=8):
    """
    用符号替代文本标签：
    - base: 圆点
    - novel: 三角形
    - background: 方形
    颜色表示不同类别
    """
    num_classes = len(labels)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))

    # 不显示默认文本
    ax.set_xticklabels([''] * num_classes)
    ax.set_yticklabels([''] * num_classes)

    # marker settings
    for idx in range(num_classes):
        if with_background and idx == num_classes - 1:
            marker = 's'
            facecolor = '#7f7f7f'
            edgecolor = 'black'
            size = 95
        else:
            if idx in base_classes:
                marker = 'o'
                size = 85
            elif idx in novel_classes:
                marker = '^'
                size = 110
            else:
                marker = 'D'
                size = 90

            facecolor = palette[idx % len(palette)]
            edgecolor = 'black'

        # x-axis markers
        ax.scatter(
            idx, -0.95,
            marker=marker,
            s=size,
            c=[facecolor],
            edgecolors=edgecolor,
            linewidths=0.8,
            clip_on=False,
            zorder=5
        )

        # y-axis markers
        ax.scatter(
            -0.95, idx,
            marker=marker,
            s=size,
            c=[facecolor],
            edgecolors=edgecolor,
            linewidths=0.8,
            clip_on=False,
            zorder=5
        )

    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=False, labeltop=False, length=0)
    ax.tick_params(axis='y', left=True, right=False, labelleft=False, labelright=False, length=0)


# def plot_confusion_matrix(confusion_matrix,
#                           labels,
#                           save_path,
#                           base_classes,
#                           novel_classes,
#                           with_background=False,
#                           show=False,
#                           title='Confusion Matrix',
#                           color_theme='Blues',
#                           normalize=False,
#                           value_fontsize=6,
#                           tick_fontsize=8,
#                           hide_title=False):
#     if normalize:
#         cm_vis = normalize_confusion_matrix(confusion_matrix)
#         value_fmt = 'pct'
#         if not hide_title:
#             title = 'Normalized ' + title
#     else:
#         cm_vis = confusion_matrix.copy()
#         value_fmt = 'int'
#
#     num_classes = len(labels)
#
#     fig_w = max(6.8, 0.52 * num_classes)
#     fig_h = max(5.8, 0.50 * num_classes)
#
#     fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
#     fig.patch.set_facecolor('white')
#     ax.set_facecolor('white')
#
#     cmap = plt.get_cmap(color_theme)
#     im = ax.imshow(cm_vis, cmap=cmap, aspect='equal', interpolation='nearest')
#
#     cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
#     cbar.ax.tick_params(labelsize=8)
#
#     if not hide_title:
#         ax.set_title(title, fontsize=11, fontweight='normal', pad=10)
#
#     ax.set_xlabel('Predicted label', fontsize=10)
#     ax.set_ylabel('Ground-truth label', fontsize=10)
#
#     import colorcet as cc
#     palette = cc.glasbey
#
#     draw_symbolic_ticklabels(
#         ax=ax,
#         labels=labels,
#         base_classes=base_classes,
#         novel_classes=novel_classes,
#         with_background=with_background,
#         palette=palette,
#         tick_fontsize=tick_fontsize
#     )
#
#     ax.xaxis.set_major_locator(MultipleLocator(1))
#     ax.yaxis.set_major_locator(MultipleLocator(1))
#     ax.xaxis.set_minor_locator(MultipleLocator(0.5))
#     ax.yaxis.set_minor_locator(MultipleLocator(0.5))
#     ax.grid(which='minor', linestyle='-', linewidth=0.6, color='white', alpha=0.8)
#     ax.tick_params(which='minor', bottom=False, left=False)
#
#     val_max = np.nanmax(cm_vis) if cm_vis.size > 0 else 0
#     thresh = val_max * 0.55
#
#     for i in range(cm_vis.shape[0]):
#         for j in range(cm_vis.shape[1]):
#             val = cm_vis[i, j]
#             if np.isnan(val):
#                 text = '0'
#                 text_color = 'black'
#             else:
#                 text = f'{val:.0f}%' if value_fmt == 'pct' else f'{int(val)}'
#                 text_color = 'white' if val >= thresh else 'black'
#
#             ax.text(
#                 j, i, text,
#                 ha='center',
#                 va='center',
#                 color=text_color,
#                 fontsize=value_fontsize)
#
#     # 给顶部和左侧符号留边距
#     ax.set_xlim(-1.4, len(labels) - 0.5)
#     ax.set_ylim(len(labels) - 0.5, -1.4)
#
#     for spine in ax.spines.values():
#         spine.set_linewidth(0.8)
#
#     fig.tight_layout()
#     plt.savefig(save_path, dpi=1200, bbox_inches='tight', facecolor='white')
#     if show:
#         plt.show()
#     plt.close(fig)

def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_path,
                          base_classes,
                          novel_classes,
                          with_background=False,
                          show=False,
                          title='Confusion Matrix',
                          color_theme='Blues',
                          normalize=False,
                          value_fontsize=8,
                          hide_title=False):
    """
    更适合单栏四小图展示的论文风格混淆矩阵：
    - 数字加粗
    - 字号更大
    - 符号标签更清晰
    - colorbar / 坐标标题加粗
    """
    if normalize:
        cm_vis = normalize_confusion_matrix(confusion_matrix)
        value_fmt = 'pct'
        if not hide_title:
            title = 'Normalized ' + title
    else:
        cm_vis = confusion_matrix.copy()
        value_fmt = 'int'

    num_classes = len(labels)

    # 单栏四图：适中比例，缩小后仍可读
    fig_w = max(6.6, 0.58 * num_classes)
    fig_h = max(5.8, 0.56 * num_classes)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(cm_vis, cmap=cmap, aspect='equal', interpolation='nearest')

    # colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cbar.ax.tick_params(labelsize=9, width=0.8)

    # colorbar 刻度字体加粗
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')

    if not hide_title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    ax.set_xlabel('Predicted label', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Ground-truth label', fontsize=11, fontweight='bold', labelpad=10)

    import colorcet as cc
    palette = cc.glasbey

    draw_symbolic_ticklabels(
        ax=ax,
        labels=labels,
        base_classes=base_classes,
        novel_classes=novel_classes,
        with_background=with_background,
        palette=palette
    )

    # 网格线
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(which='minor', linestyle='-', linewidth=0.75, color='white', alpha=0.9)
    ax.tick_params(which='minor', bottom=False, left=False)

    val_max = np.nanmax(cm_vis) if cm_vis.size > 0 else 0
    thresh = val_max * 0.55

    # 矩阵数字：加粗、增大
    for i in range(cm_vis.shape[0]):
        for j in range(cm_vis.shape[1]):
            val = cm_vis[i, j]
            if np.isnan(val):
                text = '0'
                text_color = 'black'
            else:
                text = f'{val:.0f}%' if value_fmt == 'pct' else f'{int(val)}'
                text_color = 'white' if val >= thresh else 'black'

            ax.text(
                j, i, text,
                ha='center',
                va='center',
                color=text_color,
                fontsize=value_fontsize,
                fontweight='bold'
            )

    # 给顶部和左侧符号留边距
    ax.set_xlim(-1.45, len(labels) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -1.45)

    # 外框更清晰
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    plt.savefig(save_path, dpi=1200, bbox_inches='tight', facecolor='white')
    if show:
        plt.show()
    plt.close(fig)


def save_confusion_matrix_txt(cm, labels, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('Confusion Matrix\n')
        f.write('Rows: Ground Truth, Columns: Prediction\n\n')
        f.write('\t' + '\t'.join(labels) + '\n')
        for i, row_name in enumerate(labels):
            row_vals = '\t'.join([str(int(x)) for x in cm[i]])
            f.write(f'{row_name}\t{row_vals}\n')


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    ensure_dir(args.save_dir)

    dataset = load_dataset(cfg)
    classes = list(get_classes(dataset))
    base_classes, novel_classes = get_base_novel_class_ids(
        dataset, NWPUV2_SPLIT, split=args.split)

    results = mmcv.load(args.prediction_path)

    cm = calculate_confusion_matrix(
        dataset=dataset,
        results=results,
        score_thr=args.score_thr,
        tp_iou_thr=args.tp_iou_thr,
        with_background=args.with_background)

    if args.with_background:
        plot_labels = classes + ['background']
    else:
        plot_labels = classes

    np.save(os.path.join(args.save_dir, 'confusion_matrix.npy'), cm)
    save_confusion_matrix_txt(
        cm, plot_labels, os.path.join(args.save_dir, 'confusion_matrix.txt'))

    plot_confusion_matrix(
        confusion_matrix=cm,
        labels=plot_labels,
        base_classes=base_classes,
        novel_classes=novel_classes,
        with_background=args.with_background,
        save_path=os.path.join(args.save_dir, 'confusion_matrix_count.png'),
        show=args.show,
        title='Confusion Matrix',
        color_theme=args.color_theme,
        normalize=False,
        hide_title=args.hide_title)

    plot_confusion_matrix(
        confusion_matrix=cm,
        labels=plot_labels,
        base_classes=base_classes,
        novel_classes=novel_classes,
        with_background=args.with_background,
        save_path=os.path.join(args.save_dir, 'confusion_matrix_norm.png'),
        show=False,
        title='Confusion Matrix',
        color_theme=args.color_theme,
        normalize=True,
        hide_title=args.hide_title)

    print(f'Saved confusion matrix to: {args.save_dir}')


if __name__ == '__main__':
    main()