# Copyright (c) OpenMMLab. All rights reserved.
from .contrastive_bbox_head import ContrastiveBBoxHead
from .cosine_sim_bbox_head import CosineSimBBoxHead, DisCosSimBBoxHead
from .meta_bbox_head import MetaBBoxHead
from .multi_relation_bbox_head import MultiRelationBBoxHead
from .two_branch_bbox_head import TwoBranchBBoxHead
from .kd_bbox_head import OurDisKDBBoxHead
from .ArcFaceShared2FCBBoxHead import ArcFaceShared2FCBBoxHead
from .kd_bbox_head import DisKDBBoxHead, FusionDisKDBBoxHead, ContrastiveDisKDBBoxHead, ArcFaceDisKDBBoxHead, \
    OldArcFaceDisKDBBoxHead, ChangeDisKDBBoxHead

from .cosine_sim_bbox_head import LearnableArcFaceBBoxHead, LearnableGroupArcFaceBBoxHead, ArcFaceBBoxHead, \
    DualMarginArcFaceBBoxHead, DualMarginArcFaceDisBBoxHead, DualMarginArcFaceKDDisBBoxHead, \
    DualMarginArcFaceOrthDisBBoxHead, DualMarginArcFaceMIDisBBoxHead, DualMarginArcFaceKDMIDisBBoxHead,\
    LearnableAlphaDualMarginArcFaceKDMIDisBBoxHead, PromptArcFaceDisBBoxHead, DualMarginArcFaceKDMIRDDisBBoxHead

__all__ = [
    'CosineSimBBoxHead', 'ContrastiveBBoxHead', 'MultiRelationBBoxHead',
    'MetaBBoxHead', 'TwoBranchBBoxHead',

    'DisKDBBoxHead', 'DisCosSimBBoxHead', 'OurDisKDBBoxHead', 'ChangeDisKDBBoxHead',
    'ArcFaceShared2FCBBoxHead',
    'FusionDisKDBBoxHead', 'ContrastiveDisKDBBoxHead', 'ArcFaceDisKDBBoxHead',
    'OldArcFaceDisKDBBoxHead', 'LearnableArcFaceBBoxHead', 'LearnableGroupArcFaceBBoxHead', 'ArcFaceBBoxHead',
    'DualMarginArcFaceBBoxHead', 'DualMarginArcFaceDisBBoxHead', 'DualMarginArcFaceKDDisBBoxHead',
    'DualMarginArcFaceOrthDisBBoxHead', 'DualMarginArcFaceMIDisBBoxHead', 'DualMarginArcFaceKDMIDisBBoxHead',
    'LearnableAlphaDualMarginArcFaceKDMIDisBBoxHead', 'PromptArcFaceDisBBoxHead', 'DualMarginArcFaceKDMIRDDisBBoxHead'
]
