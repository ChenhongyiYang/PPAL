import torch
import numpy as np

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.retinanet import RetinaNet
from mmdet.ppal.models.utils import bbox2result_with_uncertainty

@DETECTORS.register_module()
class ALRetinaNet(RetinaNet):

    def __init__(self,
                 **kwargs):
        super(ALRetinaNet, self).__init__(**kwargs)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.
        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result_with_uncertainty(det_bboxes, det_labels, cls_uncertainties, box_uncertainties, self.bbox_head.num_classes)
            for det_bboxes, det_labels, cls_uncertainties, box_uncertainties in results_list
        ]
        return bbox_results


