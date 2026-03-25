_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_dior.py',
    '../../../_base_/schedules/schedule.py', '../../tfa_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotDIORDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_10SHOT')],
        num_novel_shots=10,
        num_base_shots=10,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=6000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(interval=2000)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    # paramwise_cfg=dict(
    #     custom_keys={'neck': dict(lr_mult=0.1)}
    #     )
    )
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(
    warmup_iters=200, step=[
        4000, 5000
    ])
runner = dict(max_iters=6000)
# model = dict(
#     # type='GeneralizedFewShotDetector',
#     pretrained='open-mmlab://detectron2/resnet101_caffe',
#     backbone=dict(depth=101,
#                   # frozen_stages=-1
#                   ),
#     neck=dict(
#             type='FPNWithAdaptiveDCA',
#             in_channels=[256, 512, 1024, 2048],
#             out_channels=256,
#             num_outs=5,
#             dca_mode='adaptive',
#             init_cfg=[
#                 dict(
#                     type='Caffe2Xavier',
#                     override=dict(type='Caffe2Xavier', name='lateral_convs')),
#                 dict(
#                     type='Caffe2Xavier',
#                     override=dict(type='Caffe2Xavier', name='fpn_convs'))
#             ]),
#     frozen_parameters=[
#         'backbone',
#         'neck',
#         'roi_head.bbox_head.base_shared_fcs'
#     ],
#     roi_head=dict(
#         bbox_head=dict(
#             type='DisKDBBoxHead',
#             # type='KDBBoxHead',
#             fusion_type=None,
#             num_classes=20,
#             loss_kd_weight=0.025,
#             base_alpha=0.5,
#             loss_bbox=dict(loss_weight=2.0),
#             loss_cls=dict(loss_weight=1.0),
#             dis_loss=dict(
#                 type='DisLoss', num_classes=20, shot=10,
#                 loss_base_margin_weight=1.0,
#                 loss_novel_margin_weight=1.0,
#                 loss_neg_margin_weight=1.0,
#                 power_weight=4.0),
#             base_cpt='work_dirs/dior/split1-AdaptiveDCA-20250915-2256/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head_original.pth',
#             init_cfg=[
#                 dict(
#                     type='Caffe2Xavier',
#                     override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
#                 dict(
#                     type='Caffe2Xavier',
#                     override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
#                 dict(
#                     type='Normal',
#                     override=dict(type='Normal', name='fc_cls', std=0.01)),
#                 dict(
#                     type='Normal',
#                     override=dict(type='Normal', name='fc_reg', std=0.001))
#             ]
#         )
#     )
# )
model = dict(
    # type='GeneralizedFewShotDetector',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    neck=dict(
        type='FPNWithAdaptiveDCA',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        dca_mode='fpn',
        init_cfg=[
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='lateral_convs')),
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='fpn_convs'))
        ]),
    frozen_parameters=[
        'backbone',
        'neck',
        'roi_head.bbox_head.base_shared_fcs'
    ],
    roi_head=dict(
        bbox_head=dict(
            # type='ArcFaceDisKDBBoxHead',  # 替换为 ArcFaceDisKDBBoxHead
            # fusion_type=None,
            # num_classes=20,
            # loss_kd_weight=0.025,
            # base_alpha=0.5,
            # # ArcFaceLoss 配置
            # # arcface_loss_cfg=dict(
            # #     type='ArcFaceLoss',
            # #     s=16.0,
            # #     margin=0.5,
            # #     loss_weight=10.0,
            # #     reduction='sum'),
            # # 分类损失
            # loss_cls=dict(
            #     # type='ArcFaceLoss',
            #     type='ArcFaceLossAdaptive',
            #     s=16.0,
            #     # margin=0.05,
            #     loss_weight=1.0,
            #     reduction='sum'
            # ),
            type='ArcFaceDisKDBBoxHead',   # <- 使用自定义 head
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=20,
            # s=16.0,               # scale
            # margin=0.01,           # arcface margin
            # easy_margin=False,
            reg_class_agnostic=False,
            # 分类loss 这里使用我们实现的 ArcFaceLoss
            loss_cls=dict(
                type='ArcFaceLoss',
                s=16.0,
                margin=0.1,
                loss_weight=1.0,
                reduction='mean',
                ignore_index=-100
            ),
            # 回归损失
            loss_bbox=dict(
                type='SmoothL1Loss',  # 或者你之前用的 loss 类型
                beta=1.0,
                loss_weight=2.0),
            # DisLoss 保留
            dis_loss=dict(
                type='DisLoss',
                num_classes=20,
                shot=10,
                loss_base_margin_weight=1.0,
                loss_novel_margin_weight=1.0,
                loss_neg_margin_weight=1.0,
                power_weight=4.0),
            base_cpt='work_dirs/dior/split1-AdaptiveDCA-20250923-1935/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head_original.pth',
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
        )
    )
)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.

load_from = ('work_dirs/dior/split1-AdaptiveDCA-20250923-1935/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head_original.pth')
work_dir = './work_dirs/dior/split1-AdaptiveDCA-20250923-1935/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_dior-split1_10shot-fine-tuning'