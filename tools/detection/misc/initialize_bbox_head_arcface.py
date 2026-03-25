import torch
import torch.nn as nn
import argparse
# from .initialize_bbox_head import COCO_NOVEL_CLASSES, COCO_BASE_CLASSES, COCO_ALL_CLASSES, COCO_IDMAP, COCO_TAR_SIZE,\
# LVIS_NOVEL_CLASSES, LVIS_BASE_CLASSES, LVIS_ALL_CLASSES, LVIS_IDMAP, LVIS_TAR_SIZE, VOC_TAR_SIZE, NWPU_TAR_SIZE
# from .initialize_bbox_head import parse_args


"""
专门针对 ArcFace bbox_head.weight 的初始化脚本
支持三种模式：
  1. random_init: 在 base 权重的基础上，扩展 novel 类别的权重并随机初始化
  2. combine: 合并 base 权重 (src1) 和 novel 权重 (src2)
  3. remove: 保留前 N 类，其余丢弃
"""


def expand_arcface_weight(param_name, tar_size, checkpoint, checkpoint2=None, args=None):
    """扩展或合并 ArcFace 权重矩阵 (roi_head.bbox_head.weight)

    注意:
        - 权重矩阵形状为 [num_classes, feat_dim]
        - 背景类固定在最后一行
        - tar_size 表示前景类数量（不包括背景）
    """

    weight_name = param_name  # 'roi_head.bbox_head.weight'
    pretrained_weight = checkpoint['state_dict'][weight_name]
    old_num_classes_with_bg, feat_dim = pretrained_weight.shape
    old_num_classes = old_num_classes_with_bg - 1  # 去掉背景
    bg_weight = pretrained_weight[-1].clone()  # 背景类权重

    # 新的类别总数 (前景 + 背景)
    new_num_classes_with_bg = tar_size + 1
    new_weight = torch.zeros(new_num_classes_with_bg, feat_dim)

    if args.method == 'random_init':
        # 保留旧类
        new_weight[:old_num_classes] = pretrained_weight[:-1]
        # 初始化新类
        nn.init.xavier_uniform_(new_weight[old_num_classes:tar_size])
        # 复制背景
        new_weight[-1] = bg_weight

        print(f"[ArcFace-init] random_init: base={old_num_classes}, new={tar_size - old_num_classes}, bg=1")

    elif args.method == 'combine':
        assert checkpoint2 is not None, "combine 模式需要提供 --src2"
        checkpoint2_weight = checkpoint2['state_dict'][weight_name]
        novel_num_classes = checkpoint2_weight.size(0) - 1  # 去掉背景

        # base 类 (旧的前景)
        new_weight[:old_num_classes] = pretrained_weight[:-1]
        # novel 类 (新的前景)
        new_weight[old_num_classes:old_num_classes + novel_num_classes] = checkpoint2_weight[:-1]
        # 背景
        new_weight[-1] = bg_weight

        print(f"[ArcFace-init] combine: base={old_num_classes}, novel={novel_num_classes}, bg=1")

    elif args.method == 'remove':
        if tar_size > old_num_classes:
            raise ValueError("remove 模式只能裁剪类别，不能扩展类别")
        # 保留前景类
        new_weight[:tar_size] = pretrained_weight[:tar_size]
        # 背景
        new_weight[-1] = bg_weight

        print(f"[ArcFace-init] remove: keep={tar_size}, drop={old_num_classes - tar_size}, bg=1")

    else:
        raise ValueError(f"ArcFace 不支持 {args.method} 方式")

    checkpoint['state_dict'][weight_name] = new_weight


def main():
    parser = argparse.ArgumentParser(description="ArcFace 权重初始化工具")
    parser.add_argument('--src1', type=str, required=True, help='base checkpoint 路径')
    parser.add_argument('--src2', type=str, default=None, help='novel checkpoint 路径 (combine 模式需要)')
    parser.add_argument('--method', type=str, required=True, choices=['random_init', 'combine', 'remove'])
    parser.add_argument('--tar_size', type=int, required=True, help='目标类别总数 (base+novel)')
    parser.add_argument('--out', type=str, required=True, help='输出 checkpoint 路径')
    args = parser.parse_args()

    # 读取 base checkpoint
    checkpoint = torch.load(args.src1, map_location='cpu')

    # 读取 novel checkpoint（如果需要）
    checkpoint2 = None
    if args.method == 'combine' and args.src2 is not None:
        checkpoint2 = torch.load(args.src2, map_location='cpu')

    # ArcFace 的分类权重参数名
    param_name = 'roi_head.bbox_head.weight'
    if param_name not in checkpoint['state_dict']:
        raise KeyError(f"{param_name} 不在 checkpoint 里，请确认模型是 ArcFaceHead")

    # 执行扩展 / 合并 / 删除
    expand_arcface_weight(param_name, args.tar_size, checkpoint, checkpoint2, args)

    # 保存新的 checkpoint
    torch.save(checkpoint, args.out)
    print(f"[ArcFace-init] 保存到 {args.out}")


if __name__ == '__main__':
    main()
