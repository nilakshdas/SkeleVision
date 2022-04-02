# Adapted from /lib/pysot/pysot/models/model_builder.py

import logging
from collections import OrderedDict
from operator import methodcaller
from pathlib import Path

import torch
from pysot.models.head import RPNS as _RPNS
from pysot.models.loss import weight_l1_loss
from pysot.models.model_builder import ModelBuilder as _ModelBuilder

from skelevision.config import cfg
from skelevision.models.head import get_keypoint_head
from skelevision.models.loss import keypoint_loss
from skelevision.models.loss import select_cross_entropy_loss
from skelevision.utils.paths import resolve_path
from skelevision.utils.weights import T_STATEDICT
from skelevision.utils.weights import rename_legacy_backbone_names
from skelevision.utils.weights import rename_lightning_ckpt_state_dict_names

if logging.getLogger("pytorch_lightning.core").hasHandlers():
    logger = logging.getLogger("pytorch_lightning.core")
else:
    logger = logging.getLogger(__name__)


def _NO_RPN(*args, **kwargs):
    logger.info("No RPN head will be instantiated")


_RPNS.update(NO_RPN=_NO_RPN)


class MTLModelBuilder(_ModelBuilder):
    def __init__(self):
        super().__init__()

        # Build keypoint head
        self.keypoint_head = (
            get_keypoint_head(cfg.KEYPOINT.TYPE, **cfg.KEYPOINT.KWARGS)
            if cfg.KEYPOINT.TYPE is not None and cfg.TRAIN.KEYPOINT_WEIGHT > 0.0
            else None
        )

        self._load_weights()

        if cfg.MODEL_BUILDER.FREEZE_BACKBONE:
            list(map(methodcaller("requires_grad_", False), self.backbone.parameters()))
            logger.info("Gradients will not be computed for backbone parameters")

    def _load_weights(self):
        if cfg.MODEL_BUILDER.WEIGHTS_PATH is not None:
            weights_path = resolve_path(cfg.MODEL_BUILDER.WEIGHTS_PATH)
            assert weights_path.suffix in {".ckpt", ".pt", ".pth"}, weights_path
            logger.info("Loading weights from %s", weights_path)

            if weights_path.suffix == ".ckpt":
                state_dict = self._load_state_dict_ckpt(weights_path)
            else:
                state_dict = self._load_state_dict_pth(weights_path)

            ignored_keys = list()
            if self.rpn_head is None:
                filtered_state_dict: T_STATEDICT = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("rpn_head."):
                        ignored_keys.append(k)
                    else:
                        filtered_state_dict[k] = v
                state_dict = filtered_state_dict
            logger.info("Ignoring keys: %s", ignored_keys)

            incompatible_keys = self.load_state_dict(state_dict, strict=False)
            assert len(incompatible_keys.unexpected_keys) == 0, incompatible_keys
            logger.info(incompatible_keys)

    def _load_state_dict_ckpt(self, path: Path) -> T_STATEDICT:
        state_dict = torch.load(path, map_location="cpu")
        state_dict = state_dict.pop("state_dict")
        return rename_lightning_ckpt_state_dict_names(state_dict)

    def _load_state_dict_pth(self, path: Path) -> T_STATEDICT:
        state_dict = torch.load(path, map_location="cpu")
        return rename_legacy_backbone_names(state_dict)

    def forward(self, data):
        """Only used in training"""
        search = data["search"]
        template = data["template"]
        label_cls = data["label_cls"]
        label_loc = data["label_loc"]
        label_loc_weight = data["label_loc_weight"]
        label_keypoints_search = data["label_keypoints_search"]
        label_keypoints_template = data["label_keypoints_template"]

        outputs = dict()
        total_loss = None

        # Get features
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        if self.rpn_head is not None:
            cls, loc = self.rpn_head(zf, xf)

            # Compute tracking loss
            cls = self.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, label_cls)
            loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

            total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss
            total_loss += cfg.TRAIN.LOC_WEIGHT * loc_loss

            outputs.update(cls_loss=cls_loss, loc_loss=loc_loss, total_loss=total_loss)

        if self.keypoint_head is not None:
            # Get keypoints and compute keypoint loss
            if cfg.MODEL_BUILDER.USE_TEMPLATE_KEYPOINTS:
                kf = zf
                label_keypoints = label_keypoints_template
            else:
                kf = xf
                label_keypoints = label_keypoints_search
            keypoints = self.keypoint_head(kf)
            kpt_loss = keypoint_loss(
                gt_keypoints=label_keypoints,
                pred_keypoint_logits=keypoints,
                **cfg.TRAIN.KEYPOINT_LOSS_KWARGS
            )
            if kpt_loss is not None:
                wkl = cfg.TRAIN.KEYPOINT_WEIGHT * kpt_loss
                total_loss = wkl if total_loss is None else total_loss + wkl
                outputs.update(keypoint_loss=kpt_loss, total_loss=total_loss)

        if cfg.MASK.MASK:
            raise NotImplementedError

        return outputs

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        keypoint = self.keypoint_head(xf) if cfg.TRACK.KEYPOINTS else None
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
            "cls": cls,
            "loc": loc,
            "keypoint": keypoint,
            "mask": mask if cfg.MASK.MASK else None,
        }
