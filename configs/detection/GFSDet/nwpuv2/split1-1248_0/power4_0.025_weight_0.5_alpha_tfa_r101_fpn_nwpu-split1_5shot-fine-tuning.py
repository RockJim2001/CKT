_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_nwpuv2_400.py',
    '../../../_base_/schedules/schedule.py', '../../tfa_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotNWPUV2DefaultDataset',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_5SHOT')],
        num_novel_shots=5,
        num_base_shots=5,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=6000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=6000)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
optimizer = dict(lr=0.001,
                 paramwise_cfg=dict(
                     custom_keys={
                         # 解冻 C4 的 Adaptive
                         'neck.lateral_convs.1.1': dict(lr_mult=1.0, decay_mult=1.0),
                         # 解冻 C5 的 Adaptive
                         'neck.lateral_convs.2.1': dict(lr_mult=1.0, decay_mult=1.0),
                         # 解冻 C6 的 Adaptive
                         'neck.lateral_convs.3.1': dict(lr_mult=1.0, decay_mult=1.0),
                         # 其他 neck 部分保持冻结
                         'neck': dict(lr_mult=0.0, decay_mult=0.0),
                         # # 给 novel 分支较高学习率
                         # 'roi_head.bbox_head.novel_shared_fcs': dict(lr_mult=2.0),
                         # 'roi_head.bbox_head.fc_cls': dict(lr_mult=2.0),  # 针对整个 fc_cls 提高 lr
                         # 'roi_head.bbox_head.margin_base': dict(lr_mult=30.0, decay_mult=0.0),
                         # 'roi_head.bbox_head.margin_novel': dict(lr_mult=30.0, decay_mult=0.0),
                     }
                 )
                 )
optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(
    warmup_iters=100, step=[
        6000,
    ])
runner = dict(max_iters=6000)
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    frozen_parameters=[
        'backbone', 'roi_head.bbox_head.base_shared_fcs',
    ],
    neck=dict(
        type='FPNWithAdaptiveDCA',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        # dca_mode='hybrid',
        dca_mode='adaptive_0',
        init_cfg=[
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='lateral_convs')),
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='fpn_convs'))
        ]
    ),
    roi_head=dict(
        bbox_head=dict(
            type='ChangeDisKDBBoxHead',
            num_classes=10,
            loss_kd_weight=0.025,
            loss_mi_weight=0.025,
            base_alpha=0.5,
            loss_bbox=dict(loss_weight=2.0),
            loss_cls=dict(
                loss_weight=1.0,
                class_weight=[1.0] * 7 + [3.0] * 3 + [0.1]
            ),
            fusion_type='ContrastiveAttentionFusion',
            # fusion_type=None,
            dis_loss=dict(
                type='DisLoss', num_classes=10, shot=5,
                loss_base_margin_weight=1.0,
                loss_novel_margin_weight=1.0,
                loss_neg_margin_weight=1.0,
                power_weight=4.0),
            base_cpt='work_dirs/nwpuv2_rep/split1-1248_0/AdaptiveResidual/tfa_r101_fpn_nwpuv2_split1_base-training/base_model_random_init_bbox_head.pth',
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001))
            ]
        )))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.

load_from = ('work_dirs/nwpuv2_rep/split1-1248_0/AdaptiveResidual/tfa_r101_fpn_nwpuv2_split1_base-training/base_model_random_init_bbox_head.pth')
work_dir = './work_dirs/nwpuv2_rep/split1-1248_0/AdaptiveResidual/tfa_r101_fpn_nwpuv2_split1_base-training/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_nwpu-split1_5shot-fine-tuning/ContrastiveAttentionFusion'
