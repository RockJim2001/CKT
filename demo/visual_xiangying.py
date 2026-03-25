# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path
from functools import partial
import os
import cv2
import torch
import numpy as np
import mmcv
from mmcv.image import imnormalize
from mmcv import Config, DictAction
from mmfewshot.utils.det_cam_visualizer import (DetAblationLayer,
                                                DetBoxScoreTarget, DetCAMModel,
                                                DetCAMVisualizer, EigenCAM,
                                                FeatmapAM, reshape_transform)
from mmfewshot.detection.models.neck.fpn_with_adaptiveDCA import FPNWithAdaptiveDCA  # noqa

try:
    from pytorch_grad_cam import (AblationCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

GRAD_FREE_METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
    # # 'scorecam': ScoreCAM, # consumes too much memory
    'featmapam': FeatmapAM
}

GRAD_BASE_METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM
}

# ✅ 新增：自定义可扩展方法组（非grad-cam类）
CUSTOM_METHOD_MAP = {
    # # 例如：感受野热力图、特征相关性可视化、特征对齐图等
    # 'erf': 'ERF_HEATMAP',  # 仅作标记，不需要类实例
    # 'fsm': 'FEATURE_SIMILARITY_MAP',
    # 'global_attn': 'GLOBAL_ATTENTION_MAP',  # 新增
    # 'dependency': 'DEPENDENCY_HEATMAP',  # 新增
    # 'longdep': 'LONG_RANGE_DEPENDENCY',  # 新增：patch-grid 长短距离依赖
    # 'long_dep': 'LONG_RANGE_DEPENDENCY',
    # 'lcm': 'LCM',
}

# ================= 新增感受野热力图部分开始 =================

ALL_METHODS = list(GRAD_FREE_METHOD_MAP.keys() | GRAD_BASE_METHOD_MAP.keys() | CUSTOM_METHOD_MAP.keys())


# ------------------- 工具函数 -------------------

def _get_module_by_name(root_module, dotted_name):
    """
    支持属性 + list/Sequential 索引访问
    例如：
        backbone.layer3[2].conv2
        neck.fpn_convs[0]
    """
    import re
    cur = root_module
    for name in dotted_name.split('.'):
        m = re.match(r'(\w+)\[(\d+)\]', name)
        if m:
            attr_name, idx = m.group(1), int(m.group(2))
            if not hasattr(cur, attr_name):
                raise AttributeError(f"Module has no attribute {attr_name} while resolving {dotted_name}")
            cur = getattr(cur, attr_name)[idx]
        else:
            if not hasattr(cur, name):
                raise AttributeError(f"Module has no attribute {name} while resolving {dotted_name}")
            cur = getattr(cur, name)
    return cur


def _preprocess_image(img_bgr, img_norm_cfg=None, device='cuda:0', resize=None):
    """BGR -> tensor + normalize"""
    img = img_bgr.copy()
    if resize is not None:
        img = cv2.resize(img, (resize[1], resize[0]))
    img = img.astype(np.float32)

    if img_norm_cfg is None:
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True)
    if img_norm_cfg.get('to_rgb', False):
        img = img[:, :, ::-1]

    mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(img_norm_cfg['std'], dtype=np.float32)

    img = imnormalize(img, mean=mean, std=std, to_rgb=False)

    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)
    tensor.requires_grad_(True)
    return tensor


def _save_heatmap(orig_img, heatmap, out_prefix):
    heat = (heatmap * 255).astype('uint8')
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.6, heat_color, 0.4, 0)
    cv2.imwrite(f"{out_prefix}_heatmap.png", heat_color)
    cv2.imwrite(f"{out_prefix}_overlay.png", overlay)
    print(f"[ERF] saved {out_prefix}_heatmap.png and overlay")

# ---------------- 主函数 ----------------

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--method',
        default='gradcam',
        help='Type of method to use, supports '
             f'{", ".join(ALL_METHODS)}.')
    parser.add_argument(
        '--target-layers',
        default=['backbone.layer3'],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
             'specify the backbone.layer3')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--topk',
        type=int,
        default=1,
        help='Topk of the predicted result to visualizer')
    parser.add_argument(
        '--max-shape',
        nargs='+',
        type=int,
        default=20,
        help='max shapes. Its purpose is to save GPU memory. '
             'The activation map is scaled and then evaluated. '
             'If set to -1, it means no scaling.')
    parser.add_argument(
        '--no-norm-in-bbox',
        action='store_true',
        help='Norm in bbox of cam image')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
             '``cam_weights*activations``')
    parser.add_argument('--out-dir', default=None, help='dir to output file')

    # Only used by AblationCAM
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch of inference of AblationCAM')
    parser.add_argument(
        '--ratio-channels-to-ablate',
        type=int,
        default=0.5,
        help='Making it much faster of AblationCAM. '
             'The parameter controls how many channels should be ablated')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--keypoints',
        nargs='+',
        type=int,
        default=None,
        help='List of keypoints in x1 y1 x2 y2 ... format. '
             'Each pair is a keypoint in original image coordinates.'
    )
    parser.add_argument(
        '--aggregate',
        type=str,
        default='average',
        choices=['average', 'individual'],
        help='Whether to average all keypoints or output individual heatmaps'
    )
    parser.add_argument(
        '--patch-grid',
        nargs=2,
        type=int,
        default=[7, 7],
        help='Patch grid (rows cols) used by longdep, e.g. --patch-grid 7 7'
    )
    parser.add_argument(
        '--center_patch',
        nargs=2,
        type=int,
        default=[4, 4],
        help='Patch grid (rows cols) used by longdep, e.g. --patch-grid 7 7'
    )

    args = parser.parse_args()
    method_lower = args.method.lower()
    if method_lower not in ALL_METHODS:
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(ALL_METHODS)}.')

    return args


def init_model_cam(args, cfg):
    model = DetCAMModel(
        cfg, args.checkpoint, args.score_thr, device=args.device)
    if args.preview_model:
        print(model.detector)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f'model.detector.{target_layer}'))
        except Exception as e:
            print(model.detector)
            raise RuntimeError('layer does not exist', e)

    extra_params = {
        'batch_size': args.batch_size,
        'ablation_layer': DetAblationLayer(),
        'ratio_channels_to_ablate': args.ratio_channels_to_ablate
    }

    method_lower = args.method.lower()

    # ✅ 新增逻辑：判断方法类别
    if method_lower in GRAD_BASE_METHOD_MAP:
        method_class = GRAD_BASE_METHOD_MAP[method_lower]
        is_need_grad = True
        assert args.no_norm_in_bbox is False, (
            'If not norm in bbox, the visualization result may not be reasonable.'
        )
    elif method_lower in GRAD_FREE_METHOD_MAP:
        method_class = GRAD_FREE_METHOD_MAP[method_lower]
        is_need_grad = False
    elif method_lower in CUSTOM_METHOD_MAP:
        # 自定义方法（如 ERF），无需初始化 DetCAMVisualizer
        return model, None
    else:
        raise ValueError(
            f'Unknown visualization method: {args.method}. '
            f'Available: {", ".join(ALL_METHODS)}'
        )

    max_shape = args.max_shape
    if not isinstance(max_shape, list):
        max_shape = [args.max_shape]
    assert len(max_shape) == 1 or len(max_shape) == 2

    det_cam_visualizer = DetCAMVisualizer(
        method_class,
        model,
        target_layers,
        reshape_transform=partial(
            reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
        is_need_grad=is_need_grad,
        extra_params=extra_params)
    return model, det_cam_visualizer


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model, det_cam_visualizer = init_model_cam(args, cfg)

    images = args.img
    if not isinstance(images, list):
        images = [images]

    for image_path in images:
        image = cv2.imread(image_path)
        model.set_input_data(image)
        result = model()[0]

        bboxes = result['bboxes'][..., :4]
        # bboxes = np.array([[0, 0, image.shape[0], image.shape[1]]])

        # # # 构造 N 个“整图 bbox”
        # orig_bboxes = result['bboxes'][..., :4]
        # num_boxes = orig_bboxes.shape[0]
        # bboxes = np.tile(
        #     np.array([[0, 0, image.shape[0], image.shape[1]]], dtype=np.float32),
        #     (num_boxes, 1)
        # )
        scores = result['bboxes'][..., 4]
        labels = result['labels']
        segms = result['segms']
        assert bboxes is not None and len(bboxes) > 0
        if args.topk > 0:
            idxs = np.argsort(-scores)
            bboxes = bboxes[idxs[:args.topk]]
            labels = labels[idxs[:args.topk]]
            if segms is not None:
                segms = segms[idxs[:args.topk]]
        bboxes = np.array([[0, 0, image.shape[0], image.shape[1]]])

        targets = [
            DetBoxScoreTarget(bboxes=bboxes, labels=labels, segms=segms)
        ]

        if args.method in GRAD_BASE_METHOD_MAP:
            model.set_return_loss(True)
            model.set_input_data(image, bboxes=bboxes, labels=labels)
            det_cam_visualizer.switch_activations_and_grads(model)

        grayscale_cam = det_cam_visualizer(
            image,
            targets=targets,
            aug_smooth=args.aug_smooth,
            eigen_smooth=args.eigen_smooth)
        image_with_bounding_boxes = det_cam_visualizer.show_cam(
            image, bboxes, labels, grayscale_cam, not args.no_norm_in_bbox)

        if args.out_dir:
            mmcv.mkdir_or_exist(args.out_dir)
            base_name = os.path.basename(image_path)
            stem, ext = os.path.splitext(base_name)

            # 假设你有一个变量 layer_name 表示可视化的层
            # 例如: layer_name = 'C4' 或 'P5' 或 'ROI'
            # layer_name = getattr(args, 'target-layers', 'unknown')  # 从 args 获取，如果没有就用 'unknown'
            layer_name = args.target_layers[0]

            # 在输出文件名中加入 method 名称和 layer 名称
            save_name = f"{stem}_{args.method.lower()}_{layer_name}{ext}"
            out_file = os.path.join(args.out_dir, save_name)

            mmcv.imwrite(image_with_bounding_boxes, out_file)
            print(f"[INFO] Saved {args.method} visualization for layer {layer_name} to: {out_file}")
        else:
            layer_name = getattr(args, 'target-layers', 'unknown')
            window_name = f"{os.path.basename(image_path)}_{args.method}_{layer_name}"
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, image_with_bounding_boxes)
            cv2.waitKey(0)

        if args.method in GRAD_BASE_METHOD_MAP:
            model.set_return_loss(False)
            det_cam_visualizer.switch_activations_and_grads(model)


if __name__ == '__main__':
    main()

# erf 可视化感受野热力图
# python demo/visual_cam.py demo/demo.jpg configs/detection/Ours/dior/split1-rep/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_dior-split1_10shot-fine-tuning.py/home/whut/Code/G-FSDet/G-FSDet-main/work_dirs/dior-rep/split1/power4_dis_tfa_r101_fpn_dior-split1_10shot-fine-tuning/latest.pth --method erf  --target-layers backbone.layer3 --device cuda:0 --out-dir save_dir


# import os
# import cv2
# import numpy as np
# import torch
#
# SAVE_DIR = "/home/whut/Code/G-FSDet/G-FSDet-main/save_dir_baseline"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
#
# def restore_img_without_pad(
#     img_tensor,
#     orig_h=400,
#     orig_w=400,
#     mean=(103.53, 116.28, 123.675),
#     std=(1.0, 1.0, 1.0),
#     to_rgb=False
# ):
#     """
#     将归一化后的 img tensor 恢复成原图，并去掉 pad
#     输入:
#         img_tensor: (1,3,H,W) 或 (3,H,W)
#     输出:
#         img_np: uint8, (orig_h, orig_w, 3), BGR格式
#     """
#     if not isinstance(img_tensor, torch.Tensor):
#         raise TypeError(f"img_tensor must be torch.Tensor, but got {type(img_tensor)}")
#
#     if img_tensor.dim() == 4:
#         if img_tensor.size(0) != 1:
#             raise ValueError(f"Expected batch size = 1, got {img_tensor.shape}")
#         img_tensor = img_tensor[0]
#     elif img_tensor.dim() != 3:
#         raise ValueError(f"Expected shape (1,3,H,W) or (3,H,W), got {img_tensor.shape}")
#
#     # CHW -> HWC
#     img = img_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
#
#     mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
#     std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
#
#     # 反归一化
#     img = img * std + mean
#
#     # 若预处理使用 to_rgb=True，则这里转回 BGR
#     if to_rgb:
#         img = img[..., ::-1]
#
#     img = np.clip(img, 0, 255).astype(np.uint8)
#
#     # 去掉右边和下边 padding
#     img = img[:orig_h, :orig_w]
#
#     return img
#
#
# def feature_to_heatmap(feat, mode='mean'):
#     """
#     将单层特征 (1,C,H,W) 转为单通道热力图 [0,1]
#     mode:
#         'mean' : 通道均值，最稳
#         'max'  : 通道最大值
#         'l2'   : 通道L2范数
#     """
#     if not isinstance(feat, torch.Tensor):
#         raise TypeError(f"feat must be torch.Tensor, but got {type(feat)}")
#     if feat.dim() != 4 or feat.size(0) != 1:
#         raise ValueError(f"Expected feat shape (1,C,H,W), got {feat.shape}")
#
#     feat = feat[0].detach().cpu()  # (C,H,W)
#
#     if mode == 'mean':
#         heatmap = feat.mean(dim=0)
#     elif mode == 'max':
#         heatmap, _ = feat.max(dim=0)
#     elif mode == 'l2':
#         heatmap = torch.sqrt((feat ** 2).sum(dim=0))
#     else:
#         raise ValueError(f"Unsupported mode: {mode}")
#
#     heatmap = heatmap.numpy()
#     heatmap = np.maximum(heatmap, 0)
#
#     if np.isnan(heatmap).any() or np.isinf(heatmap).any():
#         heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
#
#     if heatmap.max() > heatmap.min():
#         heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
#     else:
#         heatmap = np.zeros_like(heatmap, dtype=np.float32)
#
#     return heatmap.astype(np.float32)
#
#
# def overlay_heatmap_on_image(img_bgr, heatmap, alpha=0.5):
#     """
#     将热力图叠加到原图上
#     输入:
#         img_bgr: uint8, (H,W,3)
#         heatmap: float32, (h,w), [0,1]
#     输出:
#         overlay: uint8, (H,W,3)
#         heatmap_color: uint8, (H,W,3)
#     """
#     H, W = img_bgr.shape[:2]
#
#     heatmap_resized = cv2.resize(heatmap, (W, H))
#     heatmap_uint8 = np.uint8(255 * heatmap_resized)
#     heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
#
#     overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
#     return overlay, heatmap_color
#
#
# def visualize_fpn_features(
#     img_tensor,
#     x_input,
#     save_dir=SAVE_DIR,
#     prefix="baseline",
#     orig_h=400,
#     orig_w=400,
#     mean=(103.53, 116.28, 123.675),
#     std=(1.0, 1.0, 1.0),
#     to_rgb=False,
#     mode='mean',
#     alpha=0.5
# ):
#     """
#     将 FPN 各层特征叠加到原图上并保存
#
#     参数:
#         img_tensor: (1,3,416,416) 或 (3,416,416)
#         x_input: tuple/list, 5层特征
#            x_input[0]=(1,256,104,104)
#            x_input[1]=(1,256,52,52)
#            x_input[2]=(1,256,26,26)
#            x_input[3]=(1,256,13,13)
#            x_input[4]=(1,256,7,7)
#     """
#     os.makedirs(save_dir, exist_ok=True)
#
#     if not isinstance(x_input, (tuple, list)):
#         raise TypeError(f"x_input must be tuple/list, got {type(x_input)}")
#     if len(x_input) != 5:
#         raise ValueError(f"Expected 5 feature levels, got {len(x_input)}")
#
#     # 1) 恢复原图并去pad
#     img_bgr = restore_img_without_pad(
#         img_tensor,
#         orig_h=orig_h,
#         orig_w=orig_w,
#         mean=mean,
#         std=std,
#         to_rgb=to_rgb
#     )
#
#     # 保存原图
#     raw_path = os.path.join(save_dir, f"{prefix}_img.jpg")
#     ok = cv2.imwrite(raw_path, img_bgr)
#     if not ok:
#         raise IOError(f"Failed to save image: {raw_path}")
#     print(f"[INFO] Saved raw image: {raw_path}")
#
#     # FPN层名
#     level_names = ["P2", "P3", "P4", "P5", "P6"]
#
#     # 2) 每层特征生成热力图并叠加
#     for feat, level_name in zip(x_input, level_names):
#         heatmap = feature_to_heatmap(feat, mode=mode)
#         overlay, heatmap_color = overlay_heatmap_on_image(img_bgr, heatmap, alpha=alpha)
#
#         heatmap_path = os.path.join(save_dir, f"{prefix}_{level_name}_heatmap.jpg")
#         overlay_path = os.path.join(save_dir, f"{prefix}_{level_name}_overlay.jpg")
#
#         ok1 = cv2.imwrite(heatmap_path, heatmap_color)
#         ok2 = cv2.imwrite(overlay_path, overlay)
#
#         if not ok1:
#             raise IOError(f"Failed to save heatmap: {heatmap_path}")
#         if not ok2:
#             raise IOError(f"Failed to save overlay: {overlay_path}")
#
#         print(f"[INFO] Saved {level_name} heatmap: {heatmap_path}")
#         print(f"[INFO] Saved {level_name} overlay: {overlay_path}")
#
# visualize_fpn_features(
#     img_tensor=img,
#     x_input=x,
#     save_dir="/home/whut/Code/G-FSDet/G-FSDet-main/save_dir_CKT/00008-new",
#     prefix="baseline_case1",
#     orig_h=400,
#     orig_w=400,
#     mean=(103.53, 116.28, 123.675),
#     std=(1.0, 1.0, 1.0),
#     to_rgb=False,
#     mode='mean',
#     alpha=0.5
# )