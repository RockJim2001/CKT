import torch

# 路径
src_ckpt = "work_dirs/dior/rep/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head_copy.pth"
save_ckpt = "work_dirs/dior/rep/tfa_r101_fpn_dior_split1_base-training/latest_recovered.pth"

# 参数
num_base = 15
num_novel = 5
feat_dim = None  # 自动从权重读
checkpoint = torch.load(src_ckpt)

state_dict = checkpoint["state_dict"]

# 处理分类层
fc_cls_w = state_dict["roi_head.bbox_head.fc_cls.weight"]
fc_cls_b = state_dict["roi_head.bbox_head.fc_cls.bias"]
feat_dim = fc_cls_w.size(1)

fc_cls_w_new = torch.cat([fc_cls_w[:num_base, :], fc_cls_w[-1:, :]], dim=0)
fc_cls_b_new = torch.cat([fc_cls_b[:num_base], fc_cls_b[-1:]], dim=0)

# 取前 base+bg
state_dict["roi_head.bbox_head.fc_cls.weight"] = fc_cls_w_new
state_dict["roi_head.bbox_head.fc_cls.bias"] = fc_cls_b_new

# 处理回归层
fc_reg_w = state_dict["roi_head.bbox_head.fc_reg.weight"]
fc_reg_b = state_dict["roi_head.bbox_head.fc_reg.bias"]

fc_reg_w_new = fc_reg_w[:num_base * 4, :]
fc_reg_b_new = fc_reg_b[:num_base * 4]

state_dict["roi_head.bbox_head.fc_reg.weight"] = fc_reg_w_new
state_dict["roi_head.bbox_head.fc_reg.bias"] = fc_reg_b_new

# 保存为 latest_recovered
torch.save(checkpoint, save_ckpt)
print(f"Recovered checkpoint saved to {save_ckpt}")
