import glob
import io
import os
from itertools import chain
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from armory.data import datasets
from PIL import Image

from skelevision.config import cfg

_DESCRIPTION = """
Human dataset from OTB.
"""

_CITATION = """
@inproceedings{wu2013online,
  title={Online object tracking: A benchmark},
  author={Wu, Yi and Lim, Jongwoo and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2411--2418},
  year={2013}
}
"""

_URL_FMT = "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/%s.zip"

_OTB_HUMAN = [
    "Basketball",
    "BlurBody",
    "Bolt",
    "Couple",
    "Crowds",
    "Diving",
    "Human3",
    "Human4",
    "Human6",
    "Human9",
    "Jump",
    "MotorRolling",
    "Singer2",
    "Skating1",
    "Skating2",
    "Skiing",
    "Walking",
    "Walking2",
    "Woman",
    "Bolt2",
    "Crossing",
    "Dancer",
    "Dancer2",
    "David3",
    "Girl2",
    "Gym",
    "Human2",
    "Human5",
    "Human7",
    "Human8",
    "Jogging",
    "MountainBike",
    "Singer1",
    "Skater",
    "Skater2",
    "Subway",
]


def generate_patch_coords(
    margin_ratio: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    assert 0 <= margin_ratio < 0.5, margin_ratio
    p_x1, p_x2 = int(margin_ratio * width), int((1 - margin_ratio) * width)
    p_y1, p_y2 = int(margin_ratio * height), int((1 - margin_ratio) * height)
    return p_x1, p_y1, p_x2, p_y2


def generate_urls() -> List[str]:
    _URL = []
    for video in _OTB_HUMAN:
        url = _URL_FMT % video
        _URL.append(url)

    return _URL


def _filter_files(filenames: List[str]) -> List[str]:
    filtered_files = []
    for filename in filenames:
        with open(filename, "r") as f:
            if f.read().strip() == "":
                print("Warning: %s is empty." % filename)
            else:
                filtered_files.append(filename)

    return filtered_files


def _slice_special_otb_videos(seq_name: str, img_list):
    if seq_name.lower() == "david":
        return img_list[300 - 1 : 770]
    elif seq_name.lower() == "football1":
        return img_list[:74]
    elif seq_name.lower() == "freeman3":
        return img_list[:460]
    elif seq_name.lower() == "freeman4":
        return img_list[:283]
    elif seq_name.lower() == "diving":
        return img_list[:215]
    else:
        return img_list


class OTB(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for otb_video_tracking_human dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        features = tfds.features.FeaturesDict(
            {
                "video": tfds.features.Video(
                    (None, None, None, 3),
                    encoding_format="jpeg",
                ),
                "bboxes": tfds.features.Sequence(
                    tfds.features.Tensor(
                        shape=[4], dtype=tf.int64
                    ),  # ground truth unormalized object bounding boxes given as [x1,y1,x2,y1]
                ),
            }
        )

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        _URLS = generate_urls()
        path = dl_manager.download_and_extract(_URLS)

        return [
            tfds.core.SplitGenerator(
                name="human",
                gen_kwargs={"path": path},
            )
        ]

    def _generate_examples(self, path):
        """Generator of examples for each split"""
        yield_id = 0

        # Each video path contains one video sequence
        for video_path in path:
            videos = os.listdir(video_path)
            assert len(videos) == 1

            # Filter ground truth bounding box path
            bbox_files = sorted(
                list(
                    chain.from_iterable(
                        glob.glob(os.path.join(video_path, s, "groundtruth*.txt"))
                        for s in videos
                    )
                )
            )

            bbox_files = _filter_files(bbox_files)

            # Filter video sequences
            video_dirs = [os.path.dirname(f) for f in bbox_files]
            video_names = [os.path.basename(d) for d in video_dirs]

            for idx, video in enumerate(video_names):
                assert video in _OTB_HUMAN

                # Get all frames in a video
                rgb_frames = glob.glob(os.path.join(video_path, video, "img/*.jpg"))
                rgb_frames.sort()

                rgb_imgs = []
                for img in rgb_frames:
                    img_rgb = Image.open(img).convert("RGB")
                    img_rgb = np.array(img_rgb)
                    rgb_imgs.append(img_rgb)

                rgb_imgs = _slice_special_otb_videos(video, rgb_imgs)

                # Deal with different delimeters
                with open(bbox_files[idx], "r") as f:
                    # format [x, y, w, h]
                    bboxes_orig = np.loadtxt(
                        io.StringIO(f.read().replace(",", " "))
                    ).astype(np.int64)
                # print(len(rgb_imgs), len(bboxes_orig))
                assert len(rgb_imgs) == len(bboxes_orig)
                assert bboxes_orig.shape[1] == 4

                # Convert to [x1, y1, x2, y2]
                bboxes_orig[:, 2] += bboxes_orig[:, 0]
                bboxes_orig[:, 3] += bboxes_orig[:, 1]

                bboxes_orig = list(bboxes_orig)

                example = {"video": rgb_imgs, "bboxes": bboxes_orig}
                yield_id = yield_id + 1

                yield yield_id, example


otb_tracking_context = datasets.VideoContext(
    x_shape=(None, None, None, 3), frame_rate=None
)


def otb_video_tracking_canonical_preprocessing(batch):
    batch = batch[:, : cfg.DATASET.OTB.ARMORY_NUM_MAX_FRAMES, :, :, ::-1]
    return datasets.canonical_variable_image_preprocess(otb_tracking_context, batch)


def otb_label_preprocessing(x, y):
    box_labels = y[:, : cfg.DATASET.OTB.ARMORY_NUM_MAX_FRAMES]
    box_array = np.squeeze(box_labels, axis=0)
    box_labels = [{"boxes": box_array}]
    return box_labels, None


def otb_patch_label_preprocessing(x: np.ndarray, y: np.ndarray):
    assert x.shape[0] == y.shape[0] == 1

    F_ = cfg.DATASET.OTB.ARMORY_NUM_MAX_FRAMES
    frames = x[0, :F_]
    box_array = y[0, :F_]
    box_labels = [{"boxes": box_array}]

    F, H, W, C = frames.shape
    assert C == 3

    # Define green screen coords
    margin_ratio = 0.1
    p_x1, p_y1, p_x2, p_y2 = generate_patch_coords(margin_ratio, W, H)
    gs_coords = np.array(
        [
            [p_x1, p_y1],  # top/left
            [p_x2, p_y1],  # top/right
            [p_x2, p_y2],  # bottom/right
            [p_x1, p_y2],  # bottom/left
        ]
    )

    # Generate masks
    masks = list()
    for f in range(F):
        mask = np.zeros((H, W, C)).astype(np.uint8)
        if f > 0:  # ensure patch is not in first frame
            ctx_pad = 25
            b_x1, b_y1, b_x2, b_y2 = box_array[f]
            c_x1, c_y1, c_x2, c_y2 = (  # add some context padding
                max(0, b_x1 - ctx_pad),
                max(0, b_y1 - ctx_pad),
                min(W, b_x2 + ctx_pad + 1),
                min(H, b_y2 + ctx_pad + 1),
            )
            mask[p_y1:p_y2, p_x1:p_x2] = 1  # set patch pixels to True
            mask[c_y1:c_y2, c_x1:c_x2] = 0  # set foreground pixels to False
        masks.append(mask)
    masks = np.stack(masks)

    patch_metadata = dict(gs_coords=gs_coords, masks=masks)
    return box_labels, patch_metadata


def otb_tracking(
    split: str = "human",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = otb_video_tracking_canonical_preprocessing,
    label_preprocessing_fn: Callable = otb_label_preprocessing,
    cache_dataset: bool = False,
    framework: str = "numpy",
    shuffle_files: bool = False,
    **kwargs
):
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the otb_tracking dataset"
        )
    if batch_size != 1:
        raise ValueError("otb_tracking batch size must be set to 1")

    return datasets._generator_from_tfds(
        dataset_name="otb:1.0.0",
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=otb_tracking_context,
        as_supervised=False,
        supervised_xy_keys=("video", "bboxes"),
        **kwargs,
    )


def otb_patch_tracking(
    split: str = "human",
    epochs: int = 1,
    batch_size: int = 1,
    dataset_dir: str = None,
    preprocessing_fn: Callable = otb_video_tracking_canonical_preprocessing,
    label_preprocessing_fn: Callable = otb_patch_label_preprocessing,
    cache_dataset: bool = False,
    framework: str = "numpy",
    shuffle_files: bool = False,
    **kwargs
):
    if "class_ids" in kwargs:
        raise ValueError(
            "Filtering by class is not supported for the otb_patch_tracking dataset"
        )
    if batch_size != 1:
        raise ValueError("otb_tracking batch size must be set to 1")

    return datasets._generator_from_tfds(
        dataset_name="otb:1.0.0",
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        dataset_dir=dataset_dir,
        preprocessing_fn=preprocessing_fn,
        label_preprocessing_fn=label_preprocessing_fn,
        cache_dataset=cache_dataset,
        framework=framework,
        shuffle_files=shuffle_files,
        context=otb_tracking_context,
        as_supervised=False,
        supervised_xy_keys=("video", "bboxes"),
        **kwargs,
    )
