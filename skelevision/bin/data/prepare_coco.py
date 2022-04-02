import argparse
from pathlib import Path
from typing import Iterator
from typing import List
from typing import Tuple

from pycocotools.coco import COCO
from tqdm import tqdm

from skelevision.data.metadata import close_registry
from skelevision.data.metadata import get_h5file_path
from skelevision.data.metadata import save_metadata_item
from skelevision.data.metadata import sequenceid2str
from skelevision.data.metadata import MetadataIndex
from skelevision.data.preprocessing import preprocess_instance
from skelevision.structures.instance import FrameInstanceItem


def _parse_coco_bbox(coco_bbox: List[float]) -> List[float]:
    x1 = float(coco_bbox[0])
    y1 = float(coco_bbox[1])
    x2 = float(x1 + coco_bbox[2])
    y2 = float(y1 + coco_bbox[3])
    return [x1, y1, x2, y2]


def _parse_coco_kpts(coco_kpts: List[float]) -> List[List[float]]:
    kpts = list()
    for i in range(0, 51, 3):
        x, y, v = map(float, coco_kpts[i : i + 3])
        kpts.append([v, x, y])
    assert len(kpts) == 17, len(kpts)
    return kpts


def iter_instances(
    data_dir: Path, coco_split: str
) -> Iterator[Tuple[Path, MetadataIndex, FrameInstanceItem]]:
    coco = COCO(data_dir / "annotations" / f"person_keypoints_{coco_split}.json")
    for sequenceid in coco.imgs:
        details = coco.loadImgs(sequenceid).pop(0)
        anno_ids = coco.getAnnIds(imgIds=details["id"], iscrowd=None)
        annos = coco.loadAnns(anno_ids)
        for trackid, anno in enumerate(annos):
            cat_id = anno["category_id"]
            is_crowd = bool(anno["iscrowd"])
            num_kpts = anno["num_keypoints"]
            if cat_id == 1 and (not is_crowd) and num_kpts > 5:
                mi = MetadataIndex(sequence_id=sequenceid, track_id=trackid, frame_id=0)
                imgpath = data_dir / coco_split / f"{sequenceid2str(sequenceid)}.jpg"
                instance = FrameInstanceItem(
                    image=str(imgpath),
                    bbox=_parse_coco_bbox(anno["bbox"]),
                    kpts=_parse_coco_kpts(anno["keypoints"]),
                )

                valid_instance = instance.bbox.area > 16
                for c in ["x1", "y1", "x2", "y2"]:
                    valid_instance = valid_instance and getattr(instance.bbox, c) >= 0.0

                if valid_instance:
                    yield imgpath, mi, instance


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=["train", "valid"], required=True)
    return parser


def main():
    args = cli().parse_args()
    output_path = get_h5file_path("coco", args.split)
    coco_split = dict(train="train2017", valid="val2017").get(args.split)
    for origpath, mi, instance in tqdm(list(iter_instances(args.data_dir, coco_split))):
        instance = preprocess_instance(instance)
        save_metadata_item(output_path, mi, original_path=origpath, instance=instance)

    close_registry()


if __name__ == "__main__":
    main()
