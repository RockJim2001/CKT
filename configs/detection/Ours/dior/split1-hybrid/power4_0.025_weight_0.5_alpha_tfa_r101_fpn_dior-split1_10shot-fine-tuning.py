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
checkpoint_config = dict(interval=6000)
optiDRzer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
optimizer = dict(lr=0.001,
                 # type='AdamW',
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
                         # 给 novel 分支较高学习率
                         # 'roi_head.bbox_head.novel_shared_fcs': dict(lr_mult=2.0),
                         # 'roi_head.bbox_head.fc_cls': dict(lr_mult=2.0),  # 针对整个 fc_cls 提高 lr
                         'roi_head.bbox_head.margin_base': dict(lr_mult=30.0, decay_mult=0.0),
                         'roi_head.bbox_head.margin_novel': dict(lr_mult=30.0, decay_mult=0.0),
                     }
                 )
                 )
lr_config = dict(
    warmup_iters=100, step=[
        6000,
    ])
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
runner = dict(max_iters=6000)
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    frozen_parameters=[
        'backbone',
        'roi_head.bbox_head.base_shared_fcs'
    ],
    backbone=dict(depth=101),
    neck=dict(
        type='FPNWithAdaptiveDCA',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        # dca_mode='hybrid',
        dca_mode='adaptive',
        init_cfg=[
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='lateral_convs')),
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='fpn_convs'))
        ]
    ),
    # roi_head=dict(
    #     type='ArcFaceRoIHead',
    #     bbox_head=dict(
    #         # num_classes=20
    #         type='DualMarginArcFaceKDDisBBoxHead',
    #         num_classes=20,
    #         margin_base=0.5,
    #         margin_novel=1.0,
    #         s=16,
    #         base_class_count=15,  # ✅ 需根据任务传入
    #         learnable_margin_base=True,
    #         learnable_margin_novel=True,
    #         base_alpha=0.5,
    #         loss_kd_weight=0.025,
    #         loss_cls=dict(
    #             type='MarginArcFaceLoss',
    #             s=16.0,
    #             margin=0.5,
    #             loss_weight=1.0  # 不起作用
    #         ),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    #         # dis_loss=dict(
    #         #     type='DisLoss', num_classes=20, shot=10,
    #         #     loss_base_margin_weight=1.0,
    #         #     loss_novel_margin_weight=1.0,
    #         #     loss_neg_margin_weight=1.0,
    #         #     power_weight=4.0),
    #         use_dis_loss=True,
    #         dis_loss_weight=1.0,  # 可调系数（0.05~0.2）
    #         dis_loss_cfg=dict(
    #             type='DisLoss',
    #             num_classes=20,
    #             shot=10,
    #             loss_base_margin_weight=1.0,
    #             loss_novel_margin_weight=1.0,
    #             loss_neg_margin_weight=1.0,
    #             reduction='sum',
    #             loss_weight=1.0,
    #             power_weight=2.0
    #         ),
    #         reg_class_agnostic=False,
    #
    #         base_cpt='work_dirs/dior-rep/split1/AdaptiveResidual/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head.pth',
    #         init_cfg=[
    #             dict(
    #                 type='Caffe2Xavier',
    #                 override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
    #             dict(
    #                 type='Caffe2Xavier',
    #                 override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
    #             dict(
    #                 type='Normal',
    #                 override=dict(type='Normal', name='fc_cls', std=0.01)),
    #             dict(
    #                 type='Normal',
    #                 override=dict(type='Normal', name='fc_reg', std=0.001))
    #         ]
    #     )
    # )
    # roi_head = dict(
    #     type='ArcFaceRoIHead',
    #     bbox_head=dict(
    #         type='DualMarginArcFaceOrthDisBBoxHead',
    #         num_classes=20,
    #         margin_base=0.5,
    #         margin_novel=1.0,
    #         s=16,
    #         base_class_count=15,
    #         learnable_margin_base=True,
    #         learnable_margin_novel=True,
    #         loss_cls=dict(
    #             type='MarginArcFaceLoss',
    #             s=16.0,
    #             margin=0.5,
    #             loss_weight=1.0
    #         ),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    #         use_dis_loss=True,
    #         dis_loss_weight=1.0,
    #         dis_loss_cfg=dict(
    #             type='DisLoss',
    #             num_classes=20,
    #             shot=10,
    #             loss_base_margin_weight=1.0,
    #             loss_novel_margin_weight=1.0,
    #             loss_neg_margin_weight=1.0,
    #             reduction='sum',
    #             loss_weight=1.0,
    #             power_weight=2.0
    #         ),
    #         base_cpt='work_dirs/dior-rep/split1/AdaptiveResidual/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head.pth',
    #         init_cfg=[
    #             dict(
    #                 type='Caffe2Xavier',
    #                 override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
    #             dict(
    #                 type='Caffe2Xavier',
    #                 override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
    #             dict(
    #                 type='Normal',
    #                 override=dict(type='Normal', name='fc_cls', std=0.01)),
    #             dict(
    #                 type='Normal',
    #                 override=dict(type='Normal', name='fc_reg', std=0.001))
    #         ],
    #         base_alpha=0.5,
    #         loss_orth_weight=0.1,  # 正交约束强度
    #         reg_class_agnostic=False,
    #     )
    # )
    # roi_head = dict(
    #     type='ArcFaceRoIHead',
    #     bbox_head=dict(
    #         type='DualMarginArcFaceMIDisBBoxHead',
    #         num_classes=20,
    #         margin_base=0.5,
    #         margin_novel=1.0,
    #         s=16,
    #         base_class_count=15,
    #         learnable_margin_base=True,
    #         learnable_margin_novel=True,
    #         base_cpt='work_dirs/dior-rep/split1/AdaptiveResidual/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head.pth',
    #         init_cfg=[
    #             dict(
    #                 type='Caffe2Xavier',
    #                 override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
    #             dict(
    #                 type='Caffe2Xavier',
    #                 override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
    #             dict(
    #                 type='Normal',
    #                 override=dict(type='Normal', name='fc_cls', std=0.01)),
    #             dict(
    #                 type='Normal',
    #                 override=dict(type='Normal', name='fc_reg', std=0.001))
    #         ],
    #         base_alpha=0.5,
    #         loss_mi_weight=0.001,
    #         temperature=2,
    #         loss_cls=dict(
    #             type='MarginArcFaceLoss',
    #             s=16.0,
    #             margin=0.5,
    #             loss_weight=1.0
    #         ),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    #         use_dis_loss=True,
    #         dis_loss_weight=1.0,
    #         dis_loss_cfg=dict(
    #             type='DisLoss',
    #             num_classes=20,
    #             shot=10,
    #             loss_base_margin_weight=1.0,
    #             loss_novel_margin_weight=1.0,
    #             loss_neg_margin_weight=1.0,
    #             reduction='sum',
    #             loss_weight=1.0,
    #             power_weight=2.0
    #         ),
    #         reg_class_agnostic=False
    #     )
    # )
    # roi_head=dict(
    #     type='ArcFaceRoIHead',
    #     bbox_head=dict(
    #         # num_classes=20
    #         type='DualMarginArcFaceKDMIDisBBoxHead',
    #         num_classes=20,
    #         margin_base=0.5,
    #         margin_novel=1.0,
    #         s=16,
    #         base_class_count=15,  # ✅ 需根据任务传入
    #         learnable_margin_base=True,
    #         learnable_margin_novel=True,
    #         base_alpha=0.5,
    #         loss_kd_weight=0.025,
    #         loss_mi_weight=0.025,
    #         temperature=0.2,
    #         loss_cls=dict(
    #             type='MarginArcFaceLoss',
    #             s=16.0,
    #             margin=0.5,
    #             loss_weight=1.0  # 不起作用
    #         ),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    #         # dis_loss=dict(
    #         #     type='DisLoss', num_classes=20, shot=10,
    #         #     loss_base_margin_weight=1.0,
    #         #     loss_novel_margin_weight=1.0,
    #         #     loss_neg_margin_weight=1.0,
    #         #     power_weight=4.0),
    #         use_dis_loss=True,
    #         dis_loss_weight=1.0,  # 可调系数（0.05~0.2）
    #         dis_loss_cfg=dict(
    #             type='DisLoss',
    #             num_classes=20,
    #             shot=10,
    #             loss_base_margin_weight=1.0,
    #             loss_novel_margin_weight=1.0,
    #             loss_neg_margin_weight=1.0,
    #             reduction='sum',
    #             loss_weight=1.0,
    #             power_weight=2.0
    #         ),
    #         reg_class_agnostic=False,
    #
    #         base_cpt='work_dirs/dior-rep/split1/AdaptiveResidual/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head.pth',
    #         init_cfg=[
    #             dict(
    #                 type='Caffe2Xavier',
    #                 override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
    #             dict(
    #                 type='Caffe2Xavier',
    #                 override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
    #             dict(
    #                 type='Normal',
    #                 override=dict(type='Normal', name='fc_cls', std=0.01)),
    #             dict(
    #                 type='Normal',
    #                 override=dict(type='Normal', name='fc_reg', std=0.001))
    #         ]
    #     )
    # )
    # roi_head = dict(
    #     type='ArcFaceRoIHead',
    #     bbox_head=dict(
    #         type='PromptArcFaceDisBBoxHead',
    #         num_classes=20,
    #         base_class_count=15,
    #         margin_base=0.5,
    #         margin_novel=1.0,
    #         s=16,
    #         learnable_margin_base=True,
    #         learnable_margin_novel=True,
    #         # dis_loss_weight=0.3,
    #         prompt_dim=128,
    #         use_prompt_mi=True,
    #         prompt_mi_weight=1.0,
    #         temperature=0.2,
    #         loss_cls=dict(type='MarginArcFaceLoss', s=16.0, margin=0.5, loss_weight=1.0),
    #         loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    #         # base_cpt='work_dirs/dior-rep/split1/AdaptiveResidual/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head.pth',
    #         # init_cfg=[
    #         #     dict(
    #         #         type='Caffe2Xavier',
    #         #         override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
    #         #     dict(
    #         #         type='Caffe2Xavier',
    #         #         override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
    #         #     dict(
    #         #         type='Normal',
    #         #         override=dict(type='Normal', name='fc_cls', std=0.01)),
    #         #     dict(
    #         #         type='Normal',
    #         #         override=dict(type='Normal', name='fc_reg', std=0.001))
    #         # ]
    #         use_dis_loss=True,
    #         dis_loss_weight=1.0,  # 可调系数（0.05~0.2）
    #         dis_loss_cfg=dict(
    #             type='DisLoss',
    #             num_classes=20,
    #             shot=10,
    #             loss_base_margin_weight=1.0,
    #             loss_novel_margin_weight=1.0,
    #             loss_neg_margin_weight=1.0,
    #             reduction='sum',
    #             loss_weight=1.0,
    #             power_weight=2.0
    #         ),
    #     )
    # )
    # 使用RD+KD+MI的方法
    # roi_head=dict(
    #     type='ArcFaceRoIHead',
    #     bbox_head=dict(
    #         type='DualMarginArcFaceKDMIRDDisBBoxHead',  # ✅ 支持KD+MI+RD多模式
    #         num_classes=20,
    #         margin_base=0.5,
    #         margin_novel=1.0,
    #         s=16,
    #         base_class_count=15,
    #         learnable_margin_base=True,
    #         learnable_margin_novel=True,
    #         base_alpha=0.0,
    #         temperature=0.2,
    #
    #         # ======== 控制Loss模式 ========
    #         use_kd=True,  # ✅ 启用KD蒸馏（传统特征对齐）
    #         use_mi=False,  # ✅ 启用MI互信息保持
    #         use_rd=False,  # ✅ 是否替换为关系蒸馏（RD）
    #         loss_kd_weight=0.025,
    #         loss_mi_weight=0.025,
    #         loss_rd_weight=0.0,  # RD模式时生效
    #         rd_mode='cross',
    #
    #         # ======== 分类与回归 ========
    #         loss_cls=dict(
    #             type='LearningMarginArcFaceLoss',
    #             s=64.0,
    #             init_margin=0.1,
    #             loss_weight=1.0,
    #             class_weight=[1.0] * 15 + [5.0] * 5 + [0.1]
    #         ),
    #         loss_bbox=dict(
    #             type='L1Loss',
    #             loss_weight=1.0
    #         ),
    #
    #         # ======== 判别性正则项 ========
    #         use_dis_loss=True,
    #         dis_loss_weight=1.0,
    #         dis_loss_cfg=dict(
    #             type='DisLoss',
    #             num_classes=20,
    #             shot=10,
    #             loss_base_margin_weight=1.0,
    #             loss_novel_margin_weight=1.0,
    #             loss_neg_margin_weight=1.0,
    #             reduction='sum',
    #             loss_weight=1.0,
    #             power_weight=2.0
    #         ),
    #
    #         reg_class_agnostic=True,
    #
    #         base_cpt='work_dirs/dior-rep/split1/AdaptiveResidual_ArcFace/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head.pth',
    #
    #         init_cfg=[
    #             dict(type='Caffe2Xavier', override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
    #             dict(type='Caffe2Xavier', override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
    #             dict(type='Normal', override=dict(type='Normal', name='fc_cls', std=0.01)),
    #             dict(type='Normal', override=dict(type='Normal', name='fc_reg', std=0.001))
    #         ]
    #     )
    # )
    roi_head=dict(
            bbox_head=dict(
                type='DisKDBBoxHead',
                num_classes=20,
                loss_kd_weight=0.025,
                base_alpha=0.0,
                loss_bbox=dict(loss_weight=2.0),
                loss_cls=dict(loss_weight=1.0, class_weight=[1.0] * 15 + [10.0] * 5 + [0.1]),
                dis_loss=dict(
                    type='DisLoss', num_classes=20, shot=10,
                    loss_base_margin_weight=1.0,
                    loss_novel_margin_weight=1.0,
                    loss_neg_margin_weight=1.0,
                    power_weight=4.0),
                base_cpt='work_dirs/dior-rep/split1/AdaptiveResidual/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head.pth',
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

load_from = (
    'work_dirs/dior-rep/split1/AdaptiveResidual_ArcFace/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head.pth')
work_dir = './work_dirs/dior-rep/split1/AdaptiveResidual_ArcFace/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_dior-split1_10shot-fine-tuning'
