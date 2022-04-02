from typing import Tuple
from typing import Union

import numpy as np
import torch
from kornia.geometry.transform import Resize

T_NUM = Union[float, int]


def bbox2corner(bbox: torch.Tensor) -> torch.Tensor:
    assert bbox.size() == (4,)
    x1, y1, x2, y2 = bbox
    return torch.stack([x1, y1, x2 - x1, y2 - y1])


def corner2bbox(corner: torch.Tensor) -> torch.Tensor:
    assert corner.size() == (4,)
    x, y, w, h = corner
    return torch.stack([x, y, x + w, y + h])


def bbox2center(bbox: torch.Tensor) -> torch.Tensor:
    assert bbox.size() == (4,)
    x1, y1, x2, y2 = bbox
    return torch.stack(
        [
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        ]
    )


def corner2center(corner: torch.Tensor) -> torch.Tensor:
    assert corner.size() == (4,)
    x, y, w, h = corner
    return torch.stack(
        [
            x + (w / 2),
            y + (h / 2),
            w,
            h,
        ]
    )


def center2bbox(center: torch.Tensor) -> torch.Tensor:
    assert center.size() == (4,)
    cx, cy, w, h = center
    return torch.stack(
        [
            cx - (w / 2),
            cy - (h / 2),
            cx + (w / 2),
            cy + (h / 2),
        ]
    )


def compute_context_size(
    bbox_size: Union[Tuple[T_NUM, T_NUM], torch.Tensor], context_amount: float
) -> Union[T_NUM, torch.Tensor]:

    sqrt = torch.sqrt if isinstance(bbox_size, torch.Tensor) else np.sqrt

    bbox_w, bbox_h = bbox_size
    p = context_amount * (bbox_w + bbox_h)
    w_z = bbox_w + p
    h_z = bbox_h + p
    return sqrt(w_z * h_z)


def crop_window(
    x: torch.Tensor,
    center_pos: Tuple[int, int],
    scale_size: int,
    output_size: int,
    pad_values: torch.Tensor,
) -> torch.Tensor:
    H, W, C = x.size()
    assert C == 3
    assert pad_values.size() == (3,)

    c = (scale_size + 1) / 2
    center_x, center_y = center_pos
    context_xmin = int(center_x - c + 0.5)
    context_ymin = int(center_y - c + 0.5)
    context_xmax = context_xmin + scale_size - 1
    context_ymax = context_ymin + scale_size - 1

    lft_pad = int(max(0, -context_xmin))
    top_pad = int(max(0, -context_ymin))
    rgt_pad = int(max(0, context_xmax - W + 1))
    bot_pad = int(max(0, context_ymax - H + 1))

    context_xmin = context_xmin + lft_pad
    context_xmax = context_xmax + lft_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # Apply padding
    if any([top_pad, rgt_pad, bot_pad, lft_pad]):
        size = (H + top_pad + bot_pad, W + lft_pad + rgt_pad, C)
        x_ = torch.ones(size, dtype=x.dtype).to(x.device) * pad_values
        x_[top_pad : top_pad + H, lft_pad : lft_pad + W, :] = x
        x = x_

    # Apply cropping
    x = x[
        context_ymin : context_ymax + 1,
        context_xmin : context_xmax + 1,
        :,
    ]
    assert x.size() == (scale_size, scale_size, C)

    x = x.permute(2, 0, 1)  # Make channels first
    x = x.unsqueeze(dim=0)  # Add batch dimension
    x = Resize(output_size)(x)  # Resize to final size
    assert x.size() == (1, C, output_size, output_size)

    return x
