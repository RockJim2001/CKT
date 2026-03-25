_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_dior.py',
    '../../../_base_/schedules/schedule.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'))
# evaluation = dict(
#     interval=36000,
#     class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
evaluation = dict(
    interval=3000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
lr_config = dict(warmup_iters=200, step=[24000, 32000])
runner = dict(max_iters=36000)
checkpoint_config = dict(interval=3000)
# model settings
model = dict(
    # type='GeneralizedFewShotDetector',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    neck=dict(
        type='FPNWithAdaptiveDCA',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        # dca_mode='hybrid',
        dca_mode='fpn',
        init_cfg=[
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='lateral_convs')),
            dict(
                type='Caffe2Xavier',
                override=dict(type='Caffe2Xavier', name='fpn_convs'))
        ]),
    roi_head=dict(
        # bbox_head=dict(num_classes=15)
        bbox_head=dict(
                type='ArcFaceShared2FCBBoxHead',   # <- 使用自定义 head
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=15,
                s=16.0,               # scale
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
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        )
    )
)
# using regular sampler can get a better base model
use_infinite_sampler = False
# work_dir = './work_dirs/dior/split-AdaptiveDCA/tfa_r101_fpn_dior_split1_base-training'
work_dir = './work_dirs/dior/split1-AdaptiveDCA-20250923-1935/tfa_r101_fpn_dior_split1_base-training'
