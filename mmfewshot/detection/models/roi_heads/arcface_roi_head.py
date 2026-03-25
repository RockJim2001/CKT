from mmdet.models import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from mmdet.core import bbox2roi
import torch


@HEADS.register_module()
class ArcFaceRoIHead(StandardRoIHead):
    """RoIHead that passes labels into MarginArcFaceBBoxHead during forward."""

    def _bbox_forward(self, x, rois, labels=None):
        """Box head forward function with optional label input."""
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        # 关键：判断bbox_head是否支持labels参数
        try:
            cls_score, bbox_pred = self.bbox_head(bbox_feats, labels=labels)
        except TypeError:
            # 若bbox_head不支持labels参数，则回退默认方式
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        bbox_results = self._bbox_forward(x, rois, labels=bbox_targets[0])

        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois, *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
