# Copyright (c) OpenMMLab. All rights reserved.
from .supervised_contrastive_loss import SupervisedContrastiveLoss
from .diss_loss import DisLoss, AdaptiveDisLoss
from .ArcFaceLoss import ArcFaceLoss
from .ArcFaceLoss import MarginArcFaceLoss, LearningMarginArcFaceLoss, ClasswiseMarginArcFaceLoss
__all__ = ['SupervisedContrastiveLoss', 'DisLoss', 'ArcFaceLoss', 'MarginArcFaceLoss', 'AdaptiveDisLoss',
           'LearningMarginArcFaceLoss', 'ClasswiseMarginArcFaceLoss']
