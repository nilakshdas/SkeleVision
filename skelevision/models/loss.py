from typing import Optional

import torch
import torch.nn.functional as F
from pysot.models.loss import get_cls_loss

from skelevision.utils.keypoint import keypoints_to_flattened_targets


def keypoint_loss(
    gt_keypoints: torch.Tensor,
    pred_keypoint_logits: torch.Tensor,
    ignore_occluded_keypoints: bool = False,
    loss_reduction: str = "mean",
) -> Optional[torch.Tensor]:
    """
    The trailing dimension of gt_keypoints is 3,
    where the value at index 0 is the visibility flag,
    and the values at index 1 & 2 are the x & y coordinates respectively.

    The visibility flag follows the COCO format and must be one of three integers:
    * v=0: not labeled (in which case x=y=0)
    * v=1: labeled but not visible
    * v=2: labeled and visible
    """
    N, K, H, W = pred_keypoint_logits.size()

    assert gt_keypoints.dim() == 3, gt_keypoints.size()
    assert gt_keypoints.size(0) == N, (gt_keypoints.size(), N)
    assert gt_keypoints.size(1) == K, (gt_keypoints.size(), K)
    assert gt_keypoints.size(2) == 3, gt_keypoints.size()

    visibility_flags_flat = gt_keypoints[..., 0].view(N * K)

    valid_keypoints_flat = (
        (visibility_flags_flat > 1)
        if ignore_occluded_keypoints
        else (visibility_flags_flat > 0)
    )

    if bool(valid_keypoints_flat.count_nonzero() == 0):
        return None

    heatmap_size = (H, W)
    pred_logits_flat = pred_keypoint_logits.view(N * K, H * W)
    gt_targets_flat = keypoints_to_flattened_targets(gt_keypoints, heatmap_size)
    assert gt_targets_flat.size(0) == (N * K), (gt_targets_flat.size(), N, K)

    loss = F.cross_entropy(
        pred_logits_flat[valid_keypoints_flat],
        gt_targets_flat[valid_keypoints_flat],
        reduction=loss_reduction,
    )
    assert not loss.isnan()
    return loss


def select_cross_entropy_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # Adapted from /lib/pysot/pysot/models/loss.py
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = torch.nonzero(label.data.eq(1)).squeeze().to(label.device)
    neg = torch.nonzero(label.data.eq(0)).squeeze().to(label.device)
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5
