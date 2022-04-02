import random
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from h5py import File as H5File

from skelevision.config import cfg
from skelevision.constants import DATA_DIR
from skelevision.structures.instance import bbox_to_list
from skelevision.structures.instance import kpts_to_list
from skelevision.structures.instance import FrameInstanceItem


_H5FILE_REGISTRY: Dict[Tuple[str, str], H5File] = dict()


def _get_h5file(filepath: Path, mode: str) -> H5File:
    assert mode in {"r", "w"}, mode

    fps = str(filepath)
    key = (fps, mode)
    if key not in _H5FILE_REGISTRY:
        if mode == "w":
            filepath.parent.mkdir(parents=True, exist_ok=True)
            if filepath.exists():
                raise FileExistsError(filepath)

        f = H5File(fps, mode, libver="latest", swmr=(mode == "r"))
        _H5FILE_REGISTRY[key] = f
    return _H5FILE_REGISTRY[key]


def get_h5file_path(dataset: str, split: str) -> Path:
    return DATA_DIR / dataset / "preprocessed" / f"{split}.h5"


def close_registry():
    for f in _H5FILE_REGISTRY.values():
        f.close()


PAD_SEQUENCE_ID = 12
PAD_TRACK_ID = 2
PAD_FRAME_ID = 6


def _id2str(id_: int, pad: int) -> str:
    return "{id_:0{pad}d}".format(id_=id_, pad=pad)


def sequenceid2str(id_: int) -> str:
    return _id2str(id_, pad=PAD_SEQUENCE_ID)


def trackid2str(id_: int) -> str:
    return _id2str(id_, pad=PAD_TRACK_ID)


def frameid2str(id_: int) -> str:
    return _id2str(id_, pad=PAD_FRAME_ID)


class MetadataIndex:
    def __init__(self, sequence_id: int, track_id: int, frame_id: int):
        self.sequence_id = int(sequence_id)
        self.track_id = int(track_id)
        self.frame_id = int(frame_id)

    def __str__(self) -> str:
        return self.serialize()

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, self.__class__):
            return False
        else:
            return self.serialize() == __o.serialize()

    def __hash__(self) -> int:
        return hash(self.serialize())

    @property
    def sequence_str(self) -> str:
        return sequenceid2str(self.sequence_id)

    @property
    def track_str(self) -> str:
        return trackid2str(self.track_id)

    @property
    def frame_str(self) -> str:
        return frameid2str(self.frame_id)

    @property
    def sequence_track_str(self) -> str:
        return f"{self.sequence_str}.{self.track_str}"

    def serialize(self) -> str:
        return f"{self.sequence_track_str}.{self.frame_str}"

    @classmethod
    def unserialize(cls, serialized: str) -> "MetadataIndex":
        sequence_str, track_str, frame_str = serialized.split(".")
        return cls(
            sequence_id=int(sequence_str),
            track_id=int(track_str),
            frame_id=int(frame_str),
        )


def save_metadata_item(
    filepath: Path,
    mi: MetadataIndex,
    *,
    original_path: Path,
    instance: FrameInstanceItem,
):
    image = instance.image
    bbox_np = np.array(bbox_to_list(instance.bbox))
    kpts_np = np.array(kpts_to_list(instance.kpts))

    h5file = _get_h5file(filepath, mode="w")
    grp = h5file.create_group(mi.serialize())

    grp.attrs["original_path"] = str(original_path)
    grp.create_dataset("image", data=image, chunks=image.shape)
    grp.create_dataset("bbox", data=bbox_np, chunks=bbox_np.shape)
    grp.create_dataset("kpts", data=kpts_np, chunks=kpts_np.shape)


def load_metadata_item(
    filepath: Path, mi: MetadataIndex
) -> Tuple[Path, FrameInstanceItem]:
    h5file = _get_h5file(filepath, mode="r")
    grp = h5file[mi.serialize()]
    image = np.array(grp["image"])
    bbox = np.array(grp["bbox"]).tolist()
    kpts = np.array(grp["kpts"]).tolist()
    original_path = Path(grp.attrs["original_path"])
    return original_path, FrameInstanceItem(image, bbox, kpts)


class TemplateStrategy(Enum):
    self = "self"
    first = "first"
    random = "random"
    previous = "previous"


class MetadataSet:
    def __init__(self, dataset: str, split: str, template_strategy: str):
        assert split in {"train", "valid"}, split
        self.frame_range = cfg.DATASET.FRAME_RANGE
        self.h5file_path = get_h5file_path(dataset, split)
        self.template_strategy = TemplateStrategy(template_strategy)

        h5file = _get_h5file(self.h5file_path, mode="r")
        self._seqtrk: Dict[str, List[MetadataIndex]] = defaultdict(list)
        self._index = [MetadataIndex.unserialize(k) for k in sorted(h5file.keys())]
        for mi in self._index:
            self._seqtrk[mi.sequence_track_str].append(mi)

    def _get_mi_template(self, mi: MetadataIndex) -> MetadataIndex:
        min_frame_id = mi.frame_id - self.frame_range
        max_frame_id = mi.frame_id + self.frame_range
        frame_choices = [
            mi2.frame_id
            for mi2 in self._seqtrk[mi.sequence_track_str]
            if (min_frame_id <= mi2.frame_id <= max_frame_id)
        ]
        if self.template_strategy is TemplateStrategy.self:
            template_frame_id = mi.frame_id
        elif self.template_strategy is TemplateStrategy.first:
            template_frame_id = 0
        elif self.template_strategy is TemplateStrategy.random:
            template_frame_id = random.choice((frame_choices))
        elif self.template_strategy is TemplateStrategy.previous:
            f = mi.frame_id - 1
            template_frame_id = f if f in frame_choices else mi.frame_id
        return MetadataIndex(mi.sequence_id, mi.track_id, template_frame_id)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[FrameInstanceItem, FrameInstanceItem]:
        mi_item = self._index[idx]
        mi_tmpl = self._get_mi_template(mi_item)
        _, instance_item = load_metadata_item(self.h5file_path, mi_item)
        _, instance_tmpl = load_metadata_item(self.h5file_path, mi_tmpl)
        return instance_item, instance_tmpl
