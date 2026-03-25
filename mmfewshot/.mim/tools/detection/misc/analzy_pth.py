import torch
from mmcv import Config
from mmdet.models import build_detector

# === 路径配置 ===
cfg_path = '/home/whut/Code/G-FSDet/G-FSDet-main/configs/detection/ETF/dior/split1/tfa_r101_fpn_dior_split1_base-training.py'
pth_path = '/home/whut/Code/G-FSDet/G-FSDet-main/work_dirs/dior/rep/tfa_r101_fpn_dior_split1_base-training/latest_recovered.pth'

# === 1. 加载配置文件 ===
cfg = Config.fromfile(cfg_path)
model_cfg = cfg.model

print("\n========== [模型配置结构摘要] ==========")
print(cfg.pretty_text)

# === 2. 构建模型 ===
model = build_detector(model_cfg)
print("\n========== [模型结构摘要] ==========")
print(model)

# === 3. 加载权重 ===
checkpoint = torch.load(pth_path, map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("\n[Missing keys]:", missing)
print("[Unexpected keys]:", unexpected)

# === 4. 参数冻结状态分析 ===
total_params = 0
trainable_params = 0
frozen_params = []

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
    else:
        frozen_params.append(name)

print("\n========== [参数统计信息] ==========")
print(f"参数总数: {total_params/1e6:.2f} M")
print(f"可训练参数数: {trainable_params/1e6:.2f} M")
print(f"冻结参数数: {(total_params - trainable_params)/1e6:.2f} M")
print(f"冻结比例: {(1 - trainable_params/total_params)*100:.2f}%")

print("\n========== [被冻结的模块示例] ==========")
for name in frozen_params[:30]:  # 只打印前30个
    print(name)
if len(frozen_params) > 30:
    print(f"... 共 {len(frozen_params)} 个冻结参数\n")

# === 5. 模块层级结构概览（仅打印主要组件） ===
print("\n========== [主要模块层级结构] ==========")
for name, module in model.named_children():
    print(f"{name}: {module.__class__.__name__}")
    for subname, submodule in module.named_children():
        print(f"  ├── {subname}: {submodule.__class__.__name__}")
    print()

# === 6. 可选: 打印某一层的参数形状 ===
print("\n========== [关键层参数形状示例] ==========")
for name, param in list(model.named_parameters())[:20]:
    print(f"{name:60s} {tuple(param.shape)} {'(frozen)' if not param.requires_grad else ''}")
