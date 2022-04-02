import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from typing import Iterator
from typing import List
from typing import Tuple

from skelevision.config import cfg
from skelevision.data.metadata import close_registry
from skelevision.data.metadata import get_h5file_path
from skelevision.data.metadata import save_metadata_item
from skelevision.data.metadata import MetadataIndex
from skelevision.data.preprocessing import preprocess_instance
from skelevision.structures.instance import FrameInstanceItem


DUMMY_KEYPOINTS: List[List[float]] = list(map(list, np.zeros((17, 3))))


def get_bbox(path: str) -> np.ndarray:
    bbox_np = np.loadtxt(path, delimiter=",").astype(float)
    assert bbox_np.shape[-1] == 4, bbox_np.shape
    bbox_np[:, 2] += bbox_np[:, 0]
    bbox_np[:, 3] += bbox_np[:, 1]
    return bbox_np


def iter_instances(
    data_dir: Path,
) -> Iterator[Tuple[Path, MetadataIndex, FrameInstanceItem]]:
    video_dirs = sorted(
        data_dir.glob("person-*"), key=lambda p: int(p.name.replace("person-", ""))
    )
    for sequenceid, video_dir in enumerate(video_dirs):
        img_list = sorted(
            (video_dir / "img").glob("*.jpg"),
            key=lambda p: int(p.name.replace(".jpg", "")),
        )

        bbox_np = get_bbox(str(video_dir / "groundtruth.txt"))
        assert bbox_np.shape[0] == len(img_list), (bbox_np.shape, len(img_list))

        for frameid, imgpath in enumerate(img_list):
            mi = MetadataIndex(sequence_id=sequenceid, track_id=0, frame_id=frameid)

            instance = FrameInstanceItem(
                image=str(imgpath),
                bbox=bbox_np[frameid],
                kpts=DUMMY_KEYPOINTS,
            )

            valid_instance = instance.bbox.area > 16
            for c in ["x1", "y1", "x2", "y2"]:
                valid_instance = valid_instance and getattr(instance.bbox, c) >= 0.0

            if valid_instance:
                yield imgpath, mi, instance


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    return parser


def main():
    args = cli().parse_args()
    train_filepath = get_h5file_path("lasot-person", "train")
    valid_filepath = get_h5file_path("lasot-person", "valid")

    train_count = defaultdict(int)
    valid_count = defaultdict(int)
    for origpath, mi, instance in tqdm(list(iter_instances(args.data_dir))):
        assert mi.track_id == 0, mi

        if train_count[mi.sequence_id] < cfg.DATASET.LASOT.NUM_TRAIN_IMGS:
            counter = train_count
            filepath = train_filepath
        elif valid_count[mi.sequence_id] < cfg.DATASET.LASOT.NUM_VALID_IMGS:
            counter = valid_count
            filepath = valid_filepath
        else:
            continue

        new_mi = MetadataIndex(
            sequence_id=mi.sequence_id,
            track_id=mi.track_id,
            frame_id=counter[mi.sequence_id],
        )

        instance = preprocess_instance(instance)
        save_metadata_item(filepath, new_mi, original_path=origpath, instance=instance)

        counter[mi.sequence_id] += 1

    close_registry()


if __name__ == "__main__":
    main()
