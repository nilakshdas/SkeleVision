import numpy as np
import imgaug.augmenters as iaa
from imgaug.random import RNG

from skelevision.config import cfg


def _rand_kernel(random_state: RNG) -> np.ndarray:
    # Adapted from /lib/pysot/pysot/datasets/augmentation.py
    size = cfg.DATASET.BLUR_SIZE
    kernel = np.zeros((size, size))
    c = int(size / 2)
    wx = random_state.random(1)[0]
    kernel[:, c] += 1.0 / size * wx
    kernel[c, :] += 1.0 / size * (1 - wx)
    return kernel


def _renormalize_image(image: np.ndarray) -> np.ndarray:
    min_val = image.min(axis=(0, 1))
    max_val = image.max(axis=(0, 1))
    rangeval = max_val - min_val
    image = (image - min_val) / rangeval
    image = image * rangeval.clip(min=0, max=255)
    return image.astype(np.uint8)


def affine(scale: float, shift: int) -> iaa.Augmenter:
    shift_range = (-shift, shift)
    scale_range = (1.0 - scale, 1.0 + scale)
    return iaa.Affine(
        translate_px=dict(x=shift_range, y=shift_range),
        scale=dict(x=scale_range, y=scale_range),
    )


def blur() -> iaa.Augmenter:
    def func_images(images, random_state, parents, hooks):
        out = list()
        for image in images:
            kernel = _rand_kernel(random_state)
            aug = iaa.Convolve(matrix=kernel)
            out.append(aug(images=[image]).pop(0))
        return out

    return iaa.Lambda(func_images=func_images)


def color() -> iaa.Augmenter:
    rgbVar = np.array(
        [
            [-0.55919361, 0.98062831, -0.41940627],
            [1.72091413, 0.19879334, -1.82968581],
            [4.64467907, 4.73710203, 4.88324118],
        ],
        dtype=np.float32,
    )

    def func_images(images, random_state, parents, hooks):
        out = list()
        for image in images:
            offset = np.dot(rgbVar, random_state.randn(3, 1))
            offset = offset.reshape(3)
            image = image - offset
            out.append(_renormalize_image(image))
        return out

    return iaa.Lambda(func_images=func_images)


def get_training_aug_image() -> iaa.Augmenter:
    return iaa.Sequential(
        [
            affine(cfg.DATASET.SEARCH.SCALE, cfg.DATASET.SEARCH.SHIFT),
            iaa.Sometimes(cfg.DATASET.SEARCH.COLOR, color()),
            iaa.Sometimes(cfg.DATASET.SEARCH.BLUR, blur()),
            iaa.Fliplr(p=cfg.DATASET.SEARCH.FLIP),
        ],
        random_order=False,
    )


def get_training_aug_template() -> iaa.Augmenter:
    return iaa.Sequential(
        [
            affine(cfg.DATASET.TEMPLATE.SCALE, cfg.DATASET.TEMPLATE.SHIFT),
            iaa.Sometimes(cfg.DATASET.TEMPLATE.COLOR, color()),
            iaa.Sometimes(cfg.DATASET.TEMPLATE.BLUR, blur()),
            iaa.Fliplr(p=cfg.DATASET.TEMPLATE.FLIP),
        ],
        random_order=False,
    )
