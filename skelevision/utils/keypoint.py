from typing import Tuple

import torch


def keypoints_to_flattened_targets(
    keypoints: torch.Tensor, heatmap_size: Tuple[int, int]
) -> torch.Tensor:
    """
    The trailing dimension of keypoints is 3,
    where the value at index 0 is the visibility flag,
    and the values at index 1 & 2 are the x & y coordinates respectively.
    """
    assert keypoints.dim() == 3, keypoints.size()
    assert keypoints.size(2) == 3, keypoints.size()

    N, K, _ = keypoints.size()
    H, W = heatmap_size

    v = keypoints[..., 0]
    x = keypoints[..., 1]
    y = keypoints[..., 2]

    vx = x[v > 0.0]
    vy = y[v > 0.0]
    assert torch.all(vx >= 0), vx.min()
    assert torch.all(vy >= 0), vy.min()
    assert torch.all(vx < W), (W, vx.max())
    assert torch.all(vy < H), (H, vy.max())

    indices = (y * W) + x
    return indices.view(N * K)
