from typing import Dict
from typing import Tuple
from typing import Optional

import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from torch.utils.data import Dataset
from pysot.datasets.anchor_target import AnchorTarget

from skelevision.config import cfg
from skelevision.data.metadata import MetadataSet
from skelevision.structures.instance import FrameInstanceItem


def _crop2center(instance: FrameInstanceItem, size: int) -> FrameInstanceItem:
    h, w, _ = instance.image.shape
    center_x = w / 2
    center_y = h / 2
    x1 = int(center_x - (size / 2))
    y1 = int(center_y - (size / 2))
    x2 = int(center_x + (size / 2))
    y2 = int(center_y + (size / 2))
    return instance.crop(BoundingBox(x1, y1, x2, y2))


def _apply_standard_image_aug(search: FrameInstanceItem) -> FrameInstanceItem:
    return _crop2center(search, size=cfg.TRAIN.SEARCH_SIZE)


def _apply_standard_template_aug(template: FrameInstanceItem) -> FrameInstanceItem:
    return _crop2center(template, size=cfg.TRAIN.EXEMPLAR_SIZE)


def _clip_kpts(kpts: np.ndarray, image: np.ndarray) -> np.ndarray:
    _, h, w = image.shape
    _, d = kpts.shape
    assert d == 3
    x = kpts[:, 1]
    y = kpts[:, 2]
    kpts[np.asarray(x < 0).nonzero()] = 0
    kpts[np.asarray(y < 0).nonzero()] = 0
    kpts[np.asarray(x >= w).nonzero()] = 0
    kpts[np.asarray(y >= h).nonzero()] = 0
    return kpts


class MTLDataset(Dataset):
    def __init__(
        self,
        dataset_split: str,
        template_strategy: str = "random",
        image_aug: Optional[iaa.Augmenter] = None,
        template_aug: Optional[iaa.Augmenter] = None,
    ):
        self._metadata_sets = list()
        for dataset in cfg.DATASET.NAMES:
            self._metadata_sets.append(
                MetadataSet(dataset, dataset_split, template_strategy)
            )

        self._idx2set: Dict[int, Tuple[int, MetadataSet]] = dict()
        for metadata_set in self._metadata_sets:
            for i in range(len(metadata_set)):
                idx = len(self._idx2set)
                self._idx2set[idx] = (i, metadata_set)

        self.image_aug = image_aug
        self.template_aug = template_aug
        self.anchor_target = AnchorTarget()
        self.output_format = cfg.DATASET.OUTPUT_FORMAT

    @property
    def _to_bgr(self) -> bool:
        return self.output_format == "BGR"

    def _getitem_internal(
        self, idx: int
    ) -> Tuple[FrameInstanceItem, FrameInstanceItem]:
        idx_, metadata_set = self._idx2set[idx]
        return metadata_set[idx_]

    def apply_image_aug(self, instance: FrameInstanceItem) -> FrameInstanceItem:
        # Apply additional image augmentation
        if self.image_aug is not None:
            instance = instance.augment(self.image_aug)

        # Apply standard image augmentation
        return _apply_standard_image_aug(instance)

    def apply_template_aug(self, template: FrameInstanceItem) -> FrameInstanceItem:
        # Apply additional template augmentation
        if self.template_aug is not None:
            template = template.augment(self.template_aug)

        # Apply standard template augmentation
        return _apply_standard_template_aug(template)

    def __len__(self) -> int:
        return len(self._idx2set)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        assert cfg.DATASET.NEG <= 0.0

        img, tpl = self._getitem_internal(idx)
        img = self.apply_image_aug(img)
        tpl = self.apply_template_aug(tpl)

        instance = img.to_numpy(channels_first=True, to_bgr=self._to_bgr)
        template = tpl.to_numpy(channels_first=True, to_bgr=self._to_bgr)

        output_size = cfg.TRAIN.OUTPUT_SIZE
        cls, delta, delta_weight, _ = self.anchor_target(instance.get("bbox"), output_size)

        x = instance.pop("x") # instance dict key x corresponds to image x
        z = template.pop("x") # template dict key x corresponds to image z
        instance_kpts = _clip_kpts(instance.pop("kpts"), image=x).astype(int)
        template_kpts = _clip_kpts(template.pop("kpts"), image=z).astype(int)

        return dict(
            search=x.astype(np.float32),
            template = z.astype(np.float32),
            bbox_search = instance.pop("bbox"),
            bbox_template = template.pop("bbox"),
            label_cls=cls,
            label_loc=delta,
            label_loc_weight=delta_weight,
            label_keypoints_search=instance_kpts,
            label_keypoints_template=template_kpts,
        )
