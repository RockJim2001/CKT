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
optimizer = dict(lr=0.001,
                 paramwise_cfg=dict(
                     custom_keys={
                         # 解冻 C4 的 Adaptive
                         'neck.lateral_convs.1.1': dict(lr_mult=1.0, decay_mult=1.0),
                         # 冻结 C5 的 Adaptive
                         'neck.lateral_convs.2.1': dict(lr_mult=1.0, decay_mult=1.0),
                         # 冻结 C6 的 Adaptive
                         'neck.lateral_convs.3.1': dict(lr_mult=1.0, decay_mult=1.0),
                         # 其他 neck 部分保持冻结
                         'neck': dict(lr_mult=0.0, decay_mult=0.0),
                     }
                 )
                 )
lr_config = dict(
    warmup_iters=100, step=[
        6000,
    ])
runner = dict(max_iters=6000)
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    frozen_parameters=[
        'backbone'
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
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,
            loss_cls=dict(
                            loss_weight=1.0,
                            class_weight=[1.0] * 15 + [5.0] * 5 + [0.1]
            ),
        )
    )
)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.

load_from = ('work_dirs/dior-rep/split1/AdaptiveResidual/tfa_r101_fpn_dior_split1_base-training/base_model_random_init_bbox_head.pth')
work_dir = './work_dirs/dior-rep/split1/AdaptiveResidual/ETF_r101_fpn_dior-split1_5shot-fine-tuning'
