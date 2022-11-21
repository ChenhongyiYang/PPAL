import torch
import torch.nn.functional as F
import numpy as np
import os

INF = 1e12


@torch.no_grad()
def get_img_score_distance_matrix_slow(
        all_labels,
        all_scores,
        all_feats,
        score_thr=0.,
        same_label=True,
        metric='cosine'):

        assert metric in ('l2', 'cosine', 'kl')

        n_images = all_labels.size(0)
        n_dets = all_labels.size(1)
        feat_dim = all_feats.size(-1)
        dets_indices = torch.arange(n_dets).to(device=all_feats.device)

        if metric == 'cosine':
            all_feats = F.normalize(all_feats, p=2, dim=-1)
            all_feats_t = all_feats.transpose(1, 2)

            all_score_valid = (all_scores > score_thr).to(dtype=all_feats.dtype)
            all_score_valid_t = all_score_valid[:, :, None].transpose(1, 2)

            all_scores_t = all_scores[:, :, None].transpose(1, 2)
            all_labels_t = all_labels[:, :, None].transpose(1, 2)

            distances = []
            for i in range(n_images):
                # torch.cuda.empty_cache()

                labels_i = all_labels[i]  # [n_dets]
                scores_valid_i = all_score_valid[i]  # [n_dets]
                scores_i = all_scores[i]  # [n_dets]
                feats_i = all_feats[i]  # [n_dets, feat_dim]

                feat_distances_i =  -1 * torch.matmul(feats_i.view(1, n_dets, feat_dim), all_feats_t) + 1 # [n_images, n_dets, n_dets]
                feat_distances_i[:,dets_indices, dets_indices] = 0  # force diag to 0, avoid numerical unstable
                score_valid = torch.matmul(scores_valid_i.view(1, n_dets, 1), all_score_valid_t)  # [n_images, n_dets, n_dets]

                if same_label:
                    labels_i = labels_i[:, None].repeat(1,n_dets) # [n_dets, n_dets]
                    label_valid = (labels_i.view(1, n_dets, n_dets) == all_labels_t).to(dtype=all_feats.dtype)
                else:
                    label_valid = torch.ones_like(score_valid)

                label_invalid = (1 - label_valid).to(dtype=torch.bool)
                score_invalid = (1 - score_valid).to(dtype=torch.bool)

                feat_distances_i[label_invalid] = 2.
                feat_distances_i[score_invalid] = INF

                feat_distances_i = feat_distances_i.min(dim=-1)[0]  # [n_images, n_dets]

                norm = (score_valid.max(dim=-1)[0] * scores_i[None, :]).sum(dim=-1) + 0.00001
                '''
                Potential BUG: 
                If no box > score_thr in both images, the algorithm fails. But this is unlikely to happen
                '''
                feat_distances_i[feat_distances_i > 2] = 0.

                feat_distances_i = feat_distances_i * scores_i[None, :]
                feat_distances_i = feat_distances_i.sum(dim=-1) / norm
                distances.append(feat_distances_i.cpu())

            feat_distance = torch.stack(distances, dim=0)
            feat_distance = 0.5 * (feat_distance + feat_distance.transpose(0, 1))
            return feat_distance

        elif metric == 'kl':
            assert not same_label

            all_score_valid = (all_scores > score_thr).to(dtype=all_feats.dtype)
            all_score_valid_t = all_score_valid[:, :, None].transpose(1, 2)

            all_scores_t = all_scores[:, :, None].transpose(1, 2)
            all_labels_t = all_labels[:, :, None].transpose(1, 2)

            distances = []
            for i in range(n_images):

                labels_i = all_labels[i]  # [n_dets]
                scores_valid_i = all_score_valid[i]  # [n_dets]
                scores_i = all_scores[i]  # [n_dets]
                feats_i = all_feats[i]  # [n_dets, feat_dim]
                feat_distances_i = []
                eps = 1e-12
                _pred = feats_i.view(1, n_dets, 1, feat_dim).repeat(1, 1, n_dets, 1)
                band_width = 20
                assert n_images % band_width == 0
                for j in range(n_images // band_width):
                    _target = all_feats[j*band_width:(j+1)*band_width].view(band_width,1, n_dets, feat_dim).repeat(1,n_dets,1,1)
                    kl = _target * ((_target+eps).log() - (_pred+eps).log())
                    feat_distances_i.append(kl.sum(dim=-1))
                feat_distances_i =  torch.cat(feat_distances_i, dim = 0)
                feat_distances_i[:, dets_indices, dets_indices] = 0  # force diag to 0, avoid numerical unstable
                score_valid = torch.matmul(scores_valid_i.view(1, n_dets, 1),
                                           all_score_valid_t)  # [n_images, n_dets, n_dets]

                if same_label:
                    labels_i = labels_i[:, None].repeat(1, n_dets)  # [n_dets, n_dets]
                    label_valid = (labels_i.view(1, n_dets, n_dets) == all_labels_t).to(dtype=all_feats.dtype)
                else:
                    label_valid = torch.ones_like(score_valid)

                label_invalid = (1 - label_valid).to(dtype=torch.bool)
                score_invalid = (1 - score_valid).to(dtype=torch.bool)

                feat_distances_i[label_invalid] = 2.
                feat_distances_i[score_invalid] = INF

                feat_distances_i = feat_distances_i.min(dim=-1)[0]  # [n_images, n_dets]
                norm = (score_valid.max(dim=-1)[0] * scores_i[None, :]).sum(dim=-1) + 0.00001
                '''
                Potential BUG: 
                If no box > score_thr in both images, the algorithm fails. But this is unlikely to happen
                '''
                feat_distances_i[feat_distances_i > 2] = 0.
                feat_distances_i = feat_distances_i * scores_i[None, :]
                feat_distances_i = feat_distances_i.sum(dim=-1) / norm
                distances.append(feat_distances_i.cpu())

            feat_distance = torch.stack(distances, dim=0)
            feat_distance = 0.5 * (feat_distance + feat_distance.transpose(0, 1))
            return feat_distance

        else:
            raise NotImplementedError
            return None


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Reference: MoCo v2
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def concat_all_sum(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.stack(tensors_gather, dim=-1).sum(dim=-1)
    return output


def get_inter_feats(lvl_feats, lvl_inds, boxes, img_shape):

    img_w, img_h = img_shape[:2]

    cx = ((0.5 * (boxes[:, 0] + boxes[:, 2]) / img_w) - 0.5) * 2
    cy = ((0.5 * (boxes[:, 1] + boxes[:, 3]) / img_h) - 0.5) * 2

    coor = torch.stack((cx, cy), dim=-1) # [n_det, 2]
    ret_feats = coor.new_full((coor.shape[0], lvl_feats[0].shape[0]), 0.)

    for l in range(len(lvl_feats)):
        mask_l = lvl_inds == l
        if mask_l.sum() == 0:
            continue

        feat_l = lvl_feats[l][None, :, :, :]   # [1, C, H, W]
        coor_l = coor[mask_l][None, None, :, :]  # [1, 1, n_det_lvl, 2]
        inter_feat = F.grid_sample(feat_l, coor_l, mode='bilinear')  # [1, C, 1, n_det_lvl]
        inter_feat = inter_feat.squeeze(dim=0).squeeze(dim=1).transpose(0, 1)  # [n_det_lvl, C]
        ret_feats[mask_l] = inter_feat

    return ret_feats


def bbox2result_with_uncertainty(bboxes, labels, cls_uncertainties, box_uncertainties, num_classes):
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            cls_uncertainties = cls_uncertainties.detach().cpu().numpy()
            box_uncertainties = box_uncertainties.detach().cpu().numpy()
            bboxes = np.concatenate((bboxes, cls_uncertainties.reshape(-1, 1), box_uncertainties.reshape(-1, 1)), axis=1)
        return [bboxes[labels == i, :] for i in range(num_classes)]
