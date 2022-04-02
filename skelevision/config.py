from pysot.core.config import cfg as _pysot_default_cfg
from yacs.config import CfgNode as CN


def _deprecate_key(key: str):
    parts = key.split(".")
    parent = _C
    while len(parts) > 1:
        parent = parent[parts.pop(0)]
    parent.pop(parts.pop())
    _C.register_deprecated_key(key)


_C = _pysot_default_cfg

# Add custom configuration here

# ------------------------------------------------------------------------ #
# Preprocessing
# ------------------------------------------------------------------------ #
_C.PREPROCESS = CN()
_C.PREPROCESS.CONTEXT_AMOUNT = 0.5
_C.PREPROCESS.EXEMPLAR_SIZE = 127
_C.PREPROCESS.CROP_SIZE = 511

# ------------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------------ #
_deprecate_key("DATASET.COCO")
_deprecate_key("DATASET.DET")
_deprecate_key("DATASET.GRAY")
_deprecate_key("DATASET.VID")
_deprecate_key("DATASET.VIDEOS_PER_EPOCH")
_deprecate_key("DATASET.YOUTUBEBB")

_C.DATASET.NAMES = ("coco", "lasot-person")

_C.DATASET.NEG = -1.0
_C.DATASET.FRAME_RANGE = 10

_C.DATASET.BLUR_SIZE = 7

_C.DATASET.OUTPUT_FORMAT = "BGR"

_C.DATASET.LASOT = CN()
_C.DATASET.LASOT.NUM_TRAIN_IMGS = 800
_C.DATASET.LASOT.NUM_VALID_IMGS = 100

_C.DATASET.OTB = CN()
_C.DATASET.OTB.ARMORY_NUM_MAX_FRAMES = 100

# ------------------------------------------------------------------------ #
# Model
# ------------------------------------------------------------------------ #
_deprecate_key("BACKBONE.LAYERS_LR")
_deprecate_key("BACKBONE.PRETRAINED")
_deprecate_key("BACKBONE.TRAIN_EPOCH")
_deprecate_key("BACKBONE.TRAIN_LAYERS")
_deprecate_key("CUDA")

_C.MODEL_BUILDER = CN()
_C.MODEL_BUILDER.WEIGHTS_PATH = None
_C.MODEL_BUILDER.FREEZE_BACKBONE = False
_C.MODEL_BUILDER.USE_TEMPLATE_KEYPOINTS = False

# ------------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------------ #
_deprecate_key("TRAIN.BASE_SIZE")
_deprecate_key("TRAIN.EPOCH")
_deprecate_key("TRAIN.GRAD_CLIP")
_deprecate_key("TRAIN.LOG_DIR")
_deprecate_key("TRAIN.LOG_GRADS")
_deprecate_key("TRAIN.LR")
_deprecate_key("TRAIN.LR_WARMUP")
_deprecate_key("TRAIN.MASK_WEIGHT")
_deprecate_key("TRAIN.PRINT_FREQ")
_deprecate_key("TRAIN.RESUME")
_deprecate_key("TRAIN.SNAPSHOT_DIR")
_deprecate_key("TRAIN.START_EPOCH")

_C.TRAIN.SEED = 42
_C.TRAIN.GAMMA = 0.8
_C.TRAIN.TEMPLATE_STRATEGY = "random"

_C.TRAIN.KEYPOINT_WEIGHT = 0.0
_C.TRAIN.KEYPOINT_LOSS_KWARGS = CN(new_allowed=True)

_C.TRAINER = CN()
_C.TRAINER.KWARGS = CN(new_allowed=True)

# Keypoint
_C.KEYPOINT = CN()
_C.KEYPOINT.TYPE = None
_C.KEYPOINT.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Inference
# ------------------------------------------------------------------------ #
_C.TRACK.KEYPOINTS = False

cfg = _C
