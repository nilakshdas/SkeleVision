import warnings
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import cv2
import imgaug as ia
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from imgaug.augmentables.kps import Keypoint


# >>> BEGIN monkey-patch
def _k__str__(self) -> str:
    return "Keypoint(v=%.1f, x=%.8f, y=%.8f)" % (self.v, self.x, self.y)


Keypoint.__str__ = _k__str__
# <<< FINISH monkey-patch


WARN_INCORRECT_VISIBILITY = False


def _float_int(x: float) -> float:
    return float(int(x))


def _set_visibility(kpt: Keypoint, v: float, bbox: BoundingBox) -> Keypoint:
    is_within_bbox_x = bool(bbox.x1 <= kpt.x <= bbox.x2)
    is_within_bbox_y = bool(bbox.y1 <= kpt.y <= bbox.y2)
    is_within_bbox = is_within_bbox_x and is_within_bbox_y
    setattr(kpt, "v", v if is_within_bbox else 0.0)
    if is_within_bbox and kpt.v < 1.0 and WARN_INCORRECT_VISIBILITY:
        warnings.warn("Visibility flag may be incorrectly set for %s" % kpt)
    if kpt.v < 1.0:
        kpt.x = 0.0
        kpt.y = 0.0
    return kpt


def bbox_to_list(bbox: BoundingBox) -> List[float]:
    return list(map(float, bbox.coords.flatten()))


def kpts_to_list(kpts: List[Keypoint]) -> List[List[float]]:
    return [[float(k.v), float(k.x), float(k.y)] for k in kpts]


class _LazyImage:
    KEEP_IN_MEMORY = False

    def __init__(self, image: Union[str, np.ndarray]):
        self._path = None if isinstance(image, np.ndarray) else image
        self._image = image if isinstance(image, np.ndarray) else None

    @property
    def ndarray(self) -> np.ndarray:
        image = (
            self._image
            if self._image is not None
            else cv2.cvtColor(cv2.imread(self._path), cv2.COLOR_BGR2RGB)
        )

        self._image = image if self.KEEP_IN_MEMORY else self._image

        return image


class FrameInstanceItem:
    def __init__(
        self, image: Union[str, np.ndarray], bbox: List[float], kpts: List[List[float]]
    ):
        self._image = _LazyImage(image)
        self.bbox = BoundingBox(*map(_float_int, bbox))
        self.kpts = [
            _set_visibility(
                Keypoint(_float_int(x), _float_int(y)), _float_int(v), self.bbox
            )
            for v, x, y in kpts
        ]

    def __str__(self) -> str:
        kptstr = "\n".join([f"{i}: {kpt}" for i, kpt in enumerate(self.kpts)])
        return (
            f"FrameInstanceItem:"
            f"\n- image:\n{self.image}"
            f"\n- bbox:\n{self.bbox}"
            f"\n- kpts:\n{kptstr}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def image(self) -> np.ndarray:
        return self._image.ndarray

    @property
    def image_annotated(self) -> np.ndarray:
        image = self.image
        image = self.bbox.draw_on_image(image, size=4)
        for k in self.kpts:
            color = (0, 255, 0) if k.v > 0.0 else (255, 0, 0)
            image = k.draw_on_image(image, size=8, color=color)
        return image

    def show(self):
        ia.imshow(self.image_annotated)

    def augment(self, aug: iaa.Augmenter) -> "FrameInstanceItem":
        aug_image_list, aug_bbox_list, aug_kpts_list = aug(
            images=[self.image], bounding_boxes=[self.bbox], keypoints=[self.kpts]
        )
        aug_image = aug_image_list.pop()
        aug_bbox = aug_bbox_list.pop()
        aug_kpts = [
            _set_visibility(aug_k, k.v, aug_bbox)
            for aug_k, k in zip(aug_kpts_list.pop(), self.kpts)
        ]
        return FrameInstanceItem(
            image=aug_image,
            bbox=bbox_to_list(aug_bbox),
            kpts=kpts_to_list(aug_kpts),
        )

    def crop(self, bbox: BoundingBox) -> "FrameInstanceItem":
        h, w, _ = self.image.shape
        crop_px = (bbox.y1, w - bbox.x2, h - bbox.y2, bbox.x1)
        return self.augment(iaa.Crop(px=crop_px, keep_size=False))

    def resize(self, h: Union[int, str], w: Union[int, str]) -> "FrameInstanceItem":
        return self.augment(iaa.Resize({"height": h, "width": w}))

    def to_numpy(
        self, channels_first: bool = False, to_bgr: bool = False
    ) -> Dict[str, np.ndarray]:
        image = self.image.copy()
        if to_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if channels_first:
            image = np.moveaxis(image, -1, 0)
        return dict(
            x=image,
            bbox=np.array(bbox_to_list(self.bbox)),
            kpts=np.array(kpts_to_list(self.kpts)),
        )
