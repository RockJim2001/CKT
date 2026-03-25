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
    # 'scorecam': ScoreCAM, # consumes too much memory
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
    # 例如：感受野热力图、特征相关性可视化、特征对齐图等
    'erf': 'ERF_HEATMAP',  # 仅作标记，不需要类实例
    'fsm': 'FEATURE_SIMILARITY_MAP',
    'global_attn': 'GLOBAL_ATTENTION_MAP',  # 新增
    'dependency': 'DEPENDENCY_HEATMAP',  # 新增
    'longdep': 'LONG_RANGE_DEPENDENCY',  # 新增：patch-grid 长短距离依赖
    'long_dep': 'LONG_RANGE_DEPENDENCY',
    'lcm': 'LCM',
}

# ALL_METHODS = list(GRAD_FREE_METHOD_MAP.keys() | GRAD_BASE_METHOD_MAP.keys())

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
def compute_long_range_dependency_map_LDA(
        model,
        target_layer_name,
        image_path,
        device='cuda:0',
        out_dir='./long_dep_out',
        resize_to=(512, 512),
        img_norm_cfg=None,
        patch_grid=(7, 7),
        center_patch=None,    # 新增：直接使用 patch 坐标
        local_radius=1,
        dist_weight_power=1.5,
        highfreq_enhance=True
):
    """
    Long-range dependency visualization (LDA Version)
    - 使用 patch grid 显示远距离依赖
    - 仅保留中心 patch → 远距离 patch 的依赖
    - 屏蔽局部 patch（local_radius）
    - 使用距离增强远距离响应
    - 可选高频增强（模拟 DCA 的高频提取）
    - 输出 overlay + grid 分块
    """
    import cv2, torch, numpy as np, mmcv, os
    from torch.nn import functional as F

    # ======================
    # 1. Read original image
    # ======================
    orig_img = cv2.imread(image_path)
    assert orig_img is not None
    H0, W0 = orig_img.shape[:2]

    # ======================
    # 2. Extract feature from model
    # ======================
    detector = getattr(model, 'detector', model)
    img_tensor = _preprocess_image(orig_img, img_norm_cfg, device, resize_to)

    module = _get_module_by_name(detector, target_layer_name)
    activations = {}

    def hook_fn(m, inp, out):
        activations['feat'] = out

    h = module.register_forward_hook(hook_fn)

    detector.eval()
    with torch.no_grad():
        try:
            _ = detector.extract_feat(img_tensor)
        except:
            _ = detector.backbone(img_tensor)

    h.remove()

    feat = activations['feat']
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]

    feat = feat[0]  # [C,H,W]
    C, Hf, Wf = feat.shape

    # ======================
    # 3. Divide into patch grid
    # ======================
    gh, gw = patch_grid
    ph = Hf // gh
    pw = Wf // gw

    patch_vecs = []
    for gy in range(gh):
        for gx in range(gw):
            patch = feat[:, gy * ph:(gy + 1) * ph, gx * pw:(gx + 1) * pw]
            patch_vecs.append(patch.mean(dim=[1, 2]))

    patch_vecs = torch.stack(patch_vecs, dim=0)  # [Npatch, C]

    # ======================
    # 4. Center patch
    # ======================
    if center_patch is not None:
        cy, cx = center_patch
        cy = int(cy)
        cx = int(cx)
        cy = max(0, min(gh - 1, cy))
        cx = max(0, min(gw - 1, cx))
    else:
        cy = gh // 2
        cx = gw // 2
    center_idx = cy * gw + cx

    center_vec = patch_vecs[center_idx]
    center_vec = center_vec / (center_vec.norm() + 1e-6)

    patch_norm = patch_vecs / (patch_vecs.norm(dim=1, keepdim=True) + 1e-6)

    # ======================
    # 5. Cosine similarity
    # ======================
    sim = patch_norm @ center_vec  # [Npatch]
    sim_map = sim.view(gh, gw)

    # ======================
    # 6. 屏蔽局部贡献 —— 只看远距离依赖
    # ======================
    mask = torch.ones_like(sim_map)
    for yy in range(gh):
        for xx in range(gw):
            if abs(yy - cy) <= local_radius and abs(xx - cx) <= local_radius:
                mask[yy, xx] = 0
    sim_map = sim_map * mask

    # ======================
    # 7. 长程增强 —— 距离越远权重越大
    # ======================
    dist_weight = np.zeros((gh, gw), dtype=np.float32)
    for yy in range(gh):
        for xx in range(gw):
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            dist_weight[yy, xx] = dist ** dist_weight_power

    sim_map = sim_map * torch.tensor(dist_weight, device=sim_map.device)

    # ======================
    # 8. 高频增强（模拟 AdaptiveDCA 的高频特性）
    # ======================
    if highfreq_enhance:
        laplace = torch.tensor([[-1,-1,-1],
                                [-1,8,-1],
                                [-1,-1,-1]], dtype=torch.float32,
                               device=sim_map.device).view(1,1,3,3)

        sim_map = sim_map.view(1,1,gh,gw)
        sim_map = F.conv2d(sim_map, laplace, padding=1)[0,0]

    # ======================
    # 9. Normalize
    # ======================
    sim_map_np = sim_map.cpu().numpy()
    sim_map_np = np.clip(sim_map_np, 0, None)
    sim_map_np /= (sim_map_np.max() + 1e-6)

    sim_map_big = cv2.resize(sim_map_np, (W0, H0))

    # ======================
    # 10. Heatmap overlay
    # ======================
    heat = (sim_map_big * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.55, heat_color, 0.45, 0)

    # ======================
    # 11. Draw grid lines
    # ======================
    cell_h = H0 / gh
    cell_w = W0 / gw

    for i in range(1, gw):
        x = int(i * cell_w)
        cv2.line(overlay, (x, 0), (x, H0), (255, 255, 255), 1)

    for i in range(1, gh):
        y = int(i * cell_h)
        cv2.line(overlay, (0, y), (W0, y), (255, 255, 255), 1)

    # highlight center patch
    x0 = int(cx * cell_w)
    y0 = int(cy * cell_h)
    x1 = int((cx + 1) * cell_w)
    y1 = int((cy + 1) * cell_h)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0,255,0), 2)

    # ======================
    # 12. Save
    # ======================
    mmcv.mkdir_or_exist(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, base + "_LDA_long_dep.png")

    cv2.imwrite(out_path, overlay)
    print(f"[LDA] Long-range dependency saved to: {out_path}")

    return sim_map_np


def compute_lcm(
        model,
        target_layer_name,
        image_path,
        device='cuda:0',
        out_dir='./lcm_out/lcm',
        resize_to=(512, 512),
        img_norm_cfg=None
):
    import torch, cv2, numpy as np, os, mmcv

    orig_img = cv2.imread(image_path)
    H0, W0 = orig_img.shape[:2]

    detector = getattr(model, 'detector', model)
    img_tensor = _preprocess_image(orig_img, img_norm_cfg, device, resize_to)

    module = _get_module_by_name(detector, target_layer_name)

    activations = {}

    # ----------- forward hook（关键改进）-----------
    def hook_fn(m, inp, out):
        # MMDetection 里很多层会返回 detach()
        # 所以必须确保 out 可求导
        if isinstance(out, (list, tuple)):
            out = out[-1]
        activations['feat'] = out
        out.retain_grad()
        return out

    h = module.register_forward_hook(hook_fn)

    # ----------- forward（不能使用 no_grad）-----------
    detector.eval()
    try:
        _ = detector.extract_feat(img_tensor)
    except:
        _ = detector.backbone(img_tensor)

    h.remove()

    feat = activations['feat']      # [1,C,H,W] or [C,H,W]

    if feat.dim() == 4:
        feat = feat[0]             # → [C,H,W]

    # 再确保梯度可用
    feat.requires_grad_(True)
    feat.retain_grad()

    # ----------- LCM2.0：LogSumExp energy -----------
    energy = torch.logsumexp(feat, dim=[1, 2]).sum()

    # ----------- backward（获取 grad）-----------
    detector.zero_grad()
    energy.backward()

    # feat.grad 现在一定存在
    grad = feat.grad
    if grad is None:
        raise RuntimeError("feat.grad is still None! Hook failed to capture graph.")

    grad = grad.abs()
    dep = grad.mean(dim=0).cpu().numpy()

    dep = (dep - dep.min()) / (dep.max() + 1e-6)
    dep = cv2.resize(dep, (W0, H0))

    mmcv.mkdir_or_exist(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_prefix = os.path.join(out_dir, base + "_lcm")
    _save_heatmap(orig_img, dep, out_prefix)

    return dep


def compute_long_range_dependency_map(
        model,
        target_layer_name,
        image_path,
        device='cuda:0',
        out_dir='./long_dep_out',
        resize_to=(512, 512),
        img_norm_cfg=None,
        patch_grid=(7, 7),
        center_patch=None,    # 新增：直接使用 patch 坐标
):
    """
    Long-range dependency visualization (Enhanced Version)
    - 将特征图划分为 patch
    - 固定选择中心 patch
    - 计算中心 patch 与所有 patch 的依赖（cosine similarity）
    - 用热力图显示依赖
    - 输出图添加 patch 网格线 + 中心 patch 高亮
    """
    import cv2, torch, numpy as np, mmcv, os

    # ======================
    # 1. Load original image
    # ======================
    orig_img = cv2.imread(image_path)
    assert orig_img is not None, f"Cannot load {image_path}"

    H0, W0 = orig_img.shape[:2]

    # ======================
    # 2. Extract features
    # ======================
    detector = getattr(model, 'detector', model)
    img_tensor = _preprocess_image(orig_img, img_norm_cfg, device, resize=resize_to)

    module = _get_module_by_name(detector, target_layer_name)
    activations = {}

    def hook_fn(m, inp, out):
        activations['feat'] = out

    handle = module.register_forward_hook(hook_fn)

    detector.eval()
    with torch.no_grad():
        try:
            _ = detector.extract_feat(img_tensor)
        except:
            _ = detector.backbone(img_tensor)

    handle.remove()

    feat = activations['feat']
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]

    B, C, Hf, Wf = feat.shape
    feat = feat[0]  # [C, H, W]

    # ======================
    # 3. Divide into patches
    # ======================
    grid_h, grid_w = patch_grid
    ph = Hf // grid_h
    pw = Wf // grid_w

    patch_vecs = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            patch = feat[:, gy * ph:(gy + 1) * ph, gx * pw:(gx + 1) * pw]
            patch_vecs.append(patch.mean(dim=[1, 2]))

    patch_vecs = torch.stack(patch_vecs, dim=0)  # [Npatch, C]
    Npatch = grid_h * grid_w

    # ======================
    # 4. Select the CENTER patch only
    # ======================
    if center_patch is not None:
        cy, cx = center_patch
        cy = int(cy)
        cx = int(cx)
        center_y = max(0, min(grid_h - 1, cy))
        center_x = max(0, min(grid_w - 1, cx))
    else:
        center_y = grid_h // 2
        center_x = grid_w // 2
    # center_y = grid_h // 2
    # center_x = grid_w // 2
    center_idx = center_y * grid_w + center_x
    center_vec = patch_vecs[center_idx]

    # ======================
    # 5. Compute cosine similarity
    # ======================
    center_norm = center_vec / (center_vec.norm() + 1e-8)
    patch_norm = patch_vecs / (patch_vecs.norm(dim=1, keepdim=True) + 1e-8)

    sim = patch_norm @ center_norm  # [Npatch]
    sim = sim - sim.min()
    sim = sim / (sim.max() + 1e-8)

    sim_map = sim.view(grid_h, grid_w).cpu().numpy()
    sim_map = cv2.resize(sim_map, (W0, H0))

    # ======================
    # 6. Produce heatmap
    # ======================
    heatmap = (sim_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.55, heatmap_color, 0.45, 0)

    # ======================
    # 7. Draw patch grid lines
    # ======================
    cell_h = H0 / grid_h
    cell_w = W0 / grid_w

    # Draw vertical lines
    for i in range(1, grid_w):
        x = int(i * cell_w)
        cv2.line(overlay, (x, 0), (x, H0), (255, 255, 255), 1)

    # Draw horizontal lines
    for i in range(1, grid_h):
        y = int(i * cell_h)
        cv2.line(overlay, (0, y), (W0, y), (255, 255, 255), 1)

    # ======================
    # 8. Highlight center patch
    # ======================
    x0 = int(center_x * cell_w)
    y0 = int(center_y * cell_h)
    x1 = int((center_x + 1) * cell_w)
    y1 = int((center_y + 1) * cell_h)

    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # ======================
    # 9. Save output
    # ======================
    mmcv.mkdir_or_exist(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, base + "_longdep_grid.png")
    cv2.imwrite(out_path, overlay)

    print(f"[OK] Long-range dependency map saved to {out_path}")

    return sim_map



def compute_dependency_heatmap(
        model,
        target_layer_name,
        image_path,
        ref_points=None,  # list of (x,y)
        device='cuda:0',
        out_dir='./dep_out',
        resize_to=(512, 512),
        img_norm_cfg=None,
        aggregate='average'
):
    import cv2, torch, mmcv, numpy as np

    orig_img = cv2.imread(image_path)
    assert orig_img is not None, f"Cannot load {image_path}"

    detector = getattr(model, 'detector', model)
    img_tensor = _preprocess_image(orig_img, img_norm_cfg, device, resize=resize_to)

    module = _get_module_by_name(detector, target_layer_name)
    activations = {}
    def hook_fn(m, inp, out):
        activations['feat'] = out
    handle = module.register_forward_hook(hook_fn)

    detector.eval()
    with torch.no_grad():
        try:
            _ = detector.extract_feat(img_tensor)
        except:
            _ = detector.backbone(img_tensor)
    handle.remove()

    feat = activations['feat']
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]

    B, C, Hf, Wf = feat.shape

    if ref_points is None:
        ref_points = [(orig_img.shape[1]//2, orig_img.shape[0]//2)]

    heatmaps = []
    for idx, (x, y) in enumerate(ref_points):
        cx = int(x / img_tensor.shape[3] * Wf)
        cy = int(y / img_tensor.shape[2] * Hf)

        ref_vec = feat[0, :, cy, cx]  # [C]
        feat_flat = feat[0].view(C, -1).T  # [H*W, C]

        ref_norm = ref_vec / (ref_vec.norm() + 1e-8)
        feat_norm = feat_flat / (feat_flat.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.matmul(feat_norm, ref_norm)  # [H*W]

        sim_map = sim.view(Hf, Wf).cpu().numpy()
        sim_map -= sim_map.min()
        sim_map /= (sim_map.max() + 1e-8)
        sim_map = cv2.resize(sim_map, (orig_img.shape[1], orig_img.shape[0]))
        heatmaps.append(sim_map)

        if aggregate == 'individual':
            mmcv.mkdir_or_exist(out_dir)
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_prefix = os.path.join(out_dir, f"{base}_dep_kp{idx}")
            _save_heatmap(orig_img, sim_map, out_prefix)

    if aggregate == 'average':
        avg_map = np.mean(np.stack(heatmaps, axis=0), axis=0)
        mmcv.mkdir_or_exist(out_dir)
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_prefix = os.path.join(out_dir, f"{base}_dep_kp_avg")
        _save_heatmap(orig_img, avg_map, out_prefix)
        return avg_map
    else:
        return heatmaps



def compute_global_attention_map(
        model,
        target_layer_name,
        image_path,
        device='cuda:0',
        out_dir='./attn_out',
        resize_to=(512, 512),
        img_norm_cfg=None,
        ref_point=None  # 可选 (x,y)
):
    """
    Global Attention Map:
    - 计算参考点与整张特征图的全局注意力分布
    """
    orig_img = cv2.imread(image_path)
    assert orig_img is not None, f"Cannot load {image_path}"

    detector = getattr(model, 'detector', model)
    img_tensor = _preprocess_image(orig_img, img_norm_cfg, device, resize=resize_to)

    # hook
    module = _get_module_by_name(detector, target_layer_name)
    activations = {}
    def hook_fn(m, inp, out):
        activations['feat'] = out
    handle = module.register_forward_hook(hook_fn)

    detector.eval()
    with torch.no_grad():
        try:
            _ = detector.extract_feat(img_tensor)
        except:
            _ = detector.backbone(img_tensor)
    handle.remove()
    assert 'feat' in activations, f"{target_layer_name} hook failed"

    feat = activations['feat']
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]

    B, C, Hf, Wf = feat.shape

    # 参考点
    if ref_point is not None:
        x, y = ref_point
        cx = int(x / img_tensor.shape[3] * Wf)
        cy = int(y / img_tensor.shape[2] * Hf)
    else:
        cx, cy = Wf // 2, Hf // 2

    ref_vec = feat[0, :, cy, cx]  # [C]
    feat_flat = feat[0].view(C, -1).T  # [H*W, C]

    # Cosine Attention
    ref_vec_norm = ref_vec / (ref_vec.norm() + 1e-8)
    feat_norm = feat_flat / (feat_flat.norm(dim=1, keepdim=True) + 1e-8)
    attn = torch.matmul(feat_norm, ref_vec_norm)  # [H*W]
    attn_map = attn.view(Hf, Wf).cpu().numpy()

    # Normalize & resize
    attn_map -= attn_map.min()
    attn_map /= (attn_map.max() + 1e-8)
    attn_map = cv2.resize(attn_map, (orig_img.shape[1], orig_img.shape[0]))

    # Save
    mmcv.mkdir_or_exist(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_prefix = os.path.join(out_dir, base + "_attn")
    _save_heatmap(orig_img, attn_map, out_prefix)

    return attn_map


def compute_feature_similarity_map(model, target_layer_name, image_path,
                                   device='cuda:0', out_dir='./fsm_out',
                                   ref_point=None, resize_to=(512, 512),
                                   img_norm_cfg=None):
    """
    Feature Similarity Map:
    - ref_point: (x,y) in original image coords, None -> center
    """

    # 1. load image
    orig_img = cv2.imread(image_path)
    assert orig_img is not None, f"Cannot load {image_path}"

    # 2. detector & preprocess
    detector = getattr(model, 'detector', model)
    img_tensor = _preprocess_image(orig_img, img_norm_cfg, device, resize=resize_to)

    # 3. hook
    module = _get_module_by_name(detector, target_layer_name)
    activations = {}

    def hook_fn(m, inp, out):
        activations['feat'] = out

    handle = module.register_forward_hook(hook_fn)

    # 4. forward
    detector.eval()
    with torch.no_grad():
        try:
            _ = detector.extract_feat(img_tensor)
        except:
            _ = detector.backbone(img_tensor)
    handle.remove()
    assert 'feat' in activations, f"{target_layer_name} hook failed"
    feat = activations['feat']
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]

    # 5. select reference point
    B, C, Hf, Wf = feat.shape
    if ref_point is not None:
        x, y = ref_point
        # scale to feature map
        cx = int(x / img_tensor.shape[3] * Wf)
        cy = int(y / img_tensor.shape[2] * Hf)
    else:
        cx, cy = Wf // 2, Hf // 2

    ref_vec = feat[0, :, cy, cx]  # [C]

    # 6. compute cosine similarity across spatial locations
    feat_reshape = feat[0].view(C, -1).T  # [H*W, C]
    ref_vec_norm = ref_vec / (ref_vec.norm() + 1e-8)
    feat_norm = feat_reshape / (feat_reshape.norm(dim=1, keepdim=True) + 1e-8)
    sim = torch.matmul(feat_norm, ref_vec_norm)  # [H*W]
    sim_map = sim.view(Hf, Wf).cpu().numpy()

    # 7. normalize & resize
    sim_map -= sim_map.min()
    sim_map /= (sim_map.max() + 1e-8)
    sim_map = cv2.resize(sim_map, (orig_img.shape[1], orig_img.shape[0]))

    # 8. save
    mmcv.mkdir_or_exist(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_prefix = os.path.join(out_dir, base + "_fsm")
    _save_heatmap(orig_img, sim_map, out_prefix)

    return sim_map


# ------------------- 主函数 -------------------

def compute_erf_heatmap(
        model,
        target_layer_name,
        image_path,
        device='cuda:0',
        out_dir='./erf_out',
        resize_to=(512, 512),
        img_norm_cfg=None,
        use_roi_center=None  # optional, (x1,y1,x2,y2)
):
    """
    改进版 ERF 计算
    - 支持多通道
    - 支持 ROI 中心点
    """
    # 1. load image
    orig_img = cv2.imread(image_path)
    assert orig_img is not None, f"Cannot load {image_path}"

    # 2. detector
    detector = getattr(model, 'detector', model)

    # 3. preprocess
    img_tensor = _preprocess_image(orig_img, img_norm_cfg, device, resize=resize_to)

    # 4. hook
    module = _get_module_by_name(detector, target_layer_name)
    activations = {}

    def hook_fn(m, inp, out):
        activations['feat'] = out

    handle = module.register_forward_hook(hook_fn)

    # 5. forward
    detector.eval()
    with torch.enable_grad():
        try:
            _ = detector.extract_feat(img_tensor)
        except Exception:
            _ = detector.backbone(img_tensor)
    handle.remove()
    assert 'feat' in activations, f"{target_layer_name} hook failed"

    feat = activations['feat']
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]

    B, C, Hf, Wf = feat.shape

    # 6. 确定目标点
    if use_roi_center is not None:
        # ROI中心点 (x1,y1,x2,y2) -> feature map 坐标
        x1, y1, x2, y2 = use_roi_center
        scale_x = Wf / img_tensor.shape[3]
        scale_y = Hf / img_tensor.shape[2]
        cx = int((x1 + x2) / 2 * scale_x)
        cy = int((y1 + y2) / 2 * scale_y)
    else:
        # 默认中心点
        cy, cx = Hf // 2, Wf // 2

    # 7. target activation: 多通道求和
    target = feat[0, :, cy, cx].sum()
    if img_tensor.grad is not None:
        img_tensor.grad.zero_()
    target.backward(retain_graph=False)

    # 8. 梯度 -> ERF
    grad = img_tensor.grad[0].detach().cpu().numpy()  # [C,H,W]
    grad_map = np.sum(grad ** 2, axis=0)
    grad_map = cv2.GaussianBlur(grad_map, (0, 0), sigmaX=3)
    grad_map -= grad_map.min()
    grad_map /= (grad_map.max() + 1e-8)

    # 9. resize to original
    grad_map = cv2.resize(grad_map, (orig_img.shape[1], orig_img.shape[0]))

    # 10. save
    mmcv.mkdir_or_exist(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_prefix = os.path.join(out_dir, base + "_erf")
    _save_heatmap(orig_img, grad_map, out_prefix)

    return grad_map


# ================= 新增感受野热力图部分开始 =================


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
        default=10,
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

    # 新增：检测是否属于自定义方法
    if args.method.lower() in CUSTOM_METHOD_MAP:
        m = args.method.lower()
        target_layer = args.target_layers[0]

        # 处理关键点输入
        ref_points = None
        if args.keypoints is not None and len(args.keypoints) % 2 == 0:
            ref_points = [(args.keypoints[i], args.keypoints[i + 1])
                          for i in range(0, len(args.keypoints), 2)]

        for image_path in images:
            print(f"[INFO] Running {m} on layer {target_layer} for {image_path}")

            if m == 'erf':
                compute_erf_heatmap(
                    model, target_layer, image_path,
                    device=args.device, out_dir=args.out_dir)

            elif m == 'fsm':
                compute_feature_similarity_map(
                    model, target_layer, image_path,
                    device=args.device, out_dir=args.out_dir)

            elif m == 'global_attn':

                compute_global_attention_map(

                    model, target_layer, image_path,

                    device=args.device, out_dir=args.out_dir)
            elif m == 'dependency':
                compute_dependency_heatmap(
                    model, target_layer, image_path,
                    device=args.device, out_dir=args.out_dir)
            elif m == 'longdep':
                # 读取 patch grid 参数（parse_args 中的 list -> tuple）
                pg = (args.patch_grid[0], args.patch_grid[1]) if hasattr(args, 'patch_grid') else (7, 7)
                # compute_long_range_dependency_map 返回 (patch_map, image_map)
                patch_map = compute_long_range_dependency_map(
                    model, target_layer, image_path,
                    device=args.device, out_dir=args.out_dir,
                    patch_grid=pg,
                    center_patch=(args.center_patch[0], args.center_patch[1])
                )
                # 可选：保存 patch-level 数值矩阵供进一步分析
                if args.out_dir:
                    import json
                    mmcv.mkdir_or_exist(args.out_dir)
                    base = os.path.splitext(os.path.basename(image_path))[0]
                    dump_path = os.path.join(args.out_dir, base + f"_longdep_{pg[0]}x{pg[1]}.npy")
                    np.save(dump_path, patch_map)
                    print(f"[INFO] Saved patch-level sim map to {dump_path}")
            elif m == 'lcm':
                compute_lcm(
                    model=model,
                    target_layer_name=target_layer,
                    image_path=image_path,
                    device=args.device,
                    out_dir=args.out_dir,
                    img_norm_cfg=cfg.get('img_norm_cfg', None)
                )
            elif m == 'long_dep':
                compute_long_range_dependency_map_LDA(
                    model, target_layer, image_path,
                    device=args.device,
                    out_dir=args.out_dir,
                    patch_grid=(args.patch_grid[0], args.patch_grid[1]),
                    center_patch=(args.center_patch[0], args.center_patch[1])
                )


        return

    for image_path in images:
        image = cv2.imread(image_path)
        model.set_input_data(image)
        result = model()[0]

        bboxes = result['bboxes'][..., :4]
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
            # 在输出文件名中加入 method 名称
            save_name = f"{stem}_{args.method.lower()}{ext}"
            out_file = os.path.join(args.out_dir, save_name)
            mmcv.imwrite(image_with_bounding_boxes, out_file)
            print(f"[INFO] Saved {args.method} visualization to: {out_file}")
        else:
            window_name = f"{os.path.basename(image_path)}_{args.method}"
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
