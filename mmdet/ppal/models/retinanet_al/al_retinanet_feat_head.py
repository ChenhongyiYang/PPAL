# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import batched_nms
from mmcv.runner import BaseModule, force_fp32, get_dist_info

from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.retina_head import RetinaHead

from mmdet.ppal.models.utils import get_img_score_distance_matrix_slow, concat_all_gather, get_inter_feats

@HEADS.register_module()
class RetinaHeadFeat(RetinaHead):
    def __init__(self, total_images, max_det, feat_dim, output_path, **kwargs):
        super(RetinaHeadFeat, self).__init__(**kwargs)

        _, world_size = get_dist_info()
        assert total_images % world_size == 0  # 8 GPUs
        self.total_images = total_images
        self.queue_length = total_images
        self.current_images = 0
        self.max_det = max_det
        self.feat_dim = feat_dim
        self.output_path = output_path

        self.register_buffer("det_label_queue", torch.zeros((self.queue_length, max_det)))
        self.register_buffer("det_score_queue", torch.zeros((self.queue_length, max_det)))
        self.register_buffer("det_feat_queue", torch.zeros((self.queue_length, max_det, feat_dim)))
        self.register_buffer("image_id_queue", torch.zeros((self.queue_length, 1), dtype=torch.int) - 1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        cls_score, bbox_pred, cls_feat = self.forward(feats)
        results_list = self.get_bboxes(
            cls_score, bbox_pred, fpn_feats=cls_feat, img_metas=img_metas, rescale=rescale)
        return results_list


    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   fpn_feats=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            fpn_feats_list = select_single_mlvl(fpn_feats, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, fpn_feats_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           fpn_feats_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)  # self.cls_out_channels = 80
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, 0, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, fpn_feats_list,
                                       img_meta, cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           mlvl_feats,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)
        img_shape = img_meta['img_shape']

        mlvl_bboxes_unscale = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes = mlvl_bboxes_unscale / mlvl_bboxes_unscale.new_tensor(img_meta['scale_factor'])
        else:
            raise NotImplementedError

        lvl_inds = torch.cat([torch.zeros_like(x)+i for (i,x) in enumerate(mlvl_scores)])
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores, mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]

            det_lvl_inds = lvl_inds[keep_idxs][:cfg.max_per_img]
            det_unscale_bboxes = mlvl_bboxes_unscale[keep_idxs][:cfg.max_per_img]
            det_feats = get_inter_feats(mlvl_feats, det_lvl_inds, det_unscale_bboxes, img_shape)

            cls_scores = det_bboxes[:, -1]
            cls_uncertainties = -1 * (cls_scores * torch.log(cls_scores+1e-10) + (1-cls_scores) * torch.log((1-cls_scores) + 1e-10))
            box_uncertainties = torch.zeros_like(cls_uncertainties)

            rank, world_size = get_dist_info()
            self.collect_det_info(img_meta, det_labels, cls_scores, det_feats)
            self.current_images += world_size

            if self.current_images >= self.total_images:
                torch.cuda.empty_cache()
                if rank == 0:
                    self.compute_al()
                else:
                    torch.cuda.synchronize()


            return det_bboxes, det_labels, cls_uncertainties, box_uncertainties
        else:
            raise NotImplementedError

    def collect_det_info(self, img_meta, det_labels, det_scores, det_feats):
        rank, world_size = get_dist_info()

        img_id = int(os.path.split(img_meta['filename'])[-1].split('.')[0])
        img_id = torch.tensor([[img_id]], dtype=torch.int, device=self.image_id_queue.device)

        collected_img_ids = concat_all_gather(img_id.reshape(1, 1))
        self.image_id_queue[self.current_images:(self.current_images + world_size)] = collected_img_ids

        collected_det_labels = concat_all_gather(det_labels.reshape(1, self.max_det))
        collected_det_scores = concat_all_gather(det_scores.reshape(1, self.max_det).contiguous())
        collected_det_feats  = concat_all_gather(det_feats.reshape(1, self.max_det, self.feat_dim).contiguous())
        self.det_label_queue[self.current_images:(self.current_images + world_size)] = collected_det_labels
        self.det_score_queue[self.current_images:(self.current_images + world_size)] = collected_det_scores
        self.det_feat_queue[self.current_images:(self.current_images + world_size)] = collected_det_feats
        return

    def compute_al(self):
        valid_inds = (self.image_id_queue >= 0).reshape(-1)
        image_id_queue = self.image_id_queue[valid_inds]

        det_label_queue = self.det_label_queue[valid_inds]
        det_score_queue = self.det_score_queue[valid_inds]
        det_feat_queue = self.det_feat_queue[valid_inds]

        img_dis_mat = get_img_score_distance_matrix_slow(
            det_label_queue, det_score_queue, det_feat_queue, score_thr=0.05)

        img_dis_mat = img_dis_mat.detach().cpu().numpy()
        img_ids = image_id_queue.detach().cpu().numpy()

        with open(self.output_path, 'wb') as fwb:
            np.save(fwb, img_dis_mat)
            np.save(fwb, img_ids)
        return



