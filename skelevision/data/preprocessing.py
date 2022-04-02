import math

import imgaug.augmenters as iaa
import numpy as np

from skelevision.config import cfg
from skelevision.structures.instance import FrameInstanceItem
from skelevision.utils.tracking import compute_context_size


def preprocess_instance(instance: FrameInstanceItem) -> FrameInstanceItem:
    img_h, img_w, _ = instance.image.shape
    bbox_w = instance.bbox.width
    bbox_h = instance.bbox.height
    mean_color = instance.image.mean(axis=(0, 1)).astype(np.uint8)

    s_z = compute_context_size(
        bbox_size=(bbox_w, bbox_h), context_amount=cfg.PREPROCESS.CONTEXT_AMOUNT
    )
    scale_z = cfg.PREPROCESS.EXEMPLAR_SIZE / s_z
    d_search = (cfg.PREPROCESS.CROP_SIZE - cfg.PREPROCESS.EXEMPLAR_SIZE) / 2
    pad = d_search / scale_z
    s_x = s_z + (2 * pad)
    half_s_x = s_x / 2

    lft_pad = math.ceil(half_s_x - instance.bbox.center_x)
    top_pad = math.ceil(half_s_x - instance.bbox.center_y)
    rgt_pad = math.ceil(half_s_x - (img_w - instance.bbox.center_x))
    bot_pad = math.ceil(half_s_x - (img_h - instance.bbox.center_y))

    instance = instance.augment(
        iaa.CropAndPad(
            px=(top_pad, rgt_pad, bot_pad, lft_pad),
            keep_size=False,
        )
    )

    if lft_pad > 0:
        instance.image[:, :lft_pad] = mean_color
    if top_pad > 0:
        instance.image[:top_pad, :] = mean_color
    if rgt_pad > 0:
        instance.image[:, lft_pad + img_w :] = mean_color
    if bot_pad > 0:
        instance.image[top_pad + img_h :, :] = mean_color

    return instance.resize(h=cfg.PREPROCESS.CROP_SIZE, w=cfg.PREPROCESS.CROP_SIZE)
