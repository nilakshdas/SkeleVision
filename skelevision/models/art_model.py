# Adapted from lib/pysot/pysot/tracker/siamrpn_tracker.py

import logging
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from art.estimators.object_tracking.object_tracker import ObjectTrackerMixin
from art.estimators.pytorch import PyTorchEstimator
from pysot.utils.anchor import Anchors

from skelevision.config import cfg
from skelevision.models.model_builder import MTLModelBuilder
from skelevision.utils.paths import resolve_path
from skelevision.utils.tracking import bbox2center
from skelevision.utils.tracking import bbox2corner
from skelevision.utils.tracking import center2bbox
from skelevision.utils.tracking import compute_context_size
from skelevision.utils.tracking import crop_window

if TYPE_CHECKING:
    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.defences.preprocessor.preprocessor import Preprocessor

T_PREPROCESSING_DEFENSE = Union["Preprocessor", List["Preprocessor"], None]
T_POSTPROCESSING_DEFENSE = Union["Postprocessor", List["Postprocessor"], None]


logger = logging.getLogger(__name__)


def _expansion_rate(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, 1 / x)


def _ctx_size(w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    assert w.size() == h.size()
    pad = (w + h) * 0.5
    return torch.sqrt((w + pad) * (h + pad))


def _clip_center(cx, cy, w, h, boundary):
    #! CAVEAT: NOT differentiable in edge conditions
    cx = torch.clamp(cx, min=0, max=boundary[1])
    cy = torch.clamp(cy, min=0, max=boundary[0])
    w = torch.clamp(w, min=10, max=boundary[1])
    h = torch.clamp(h, min=10, max=boundary[0])
    return cx, cy, w, h


class ARTObjectTracker(ObjectTrackerMixin, PyTorchEstimator):
    def __init__(
        self,
        preprocessing_defences: T_PREPROCESSING_DEFENSE = None,
        postprocessing_defences: T_POSTPROCESSING_DEFENSE = None,
        device_type: str = "gpu",
    ):
        super().__init__(
            model=MTLModelBuilder(),
            clip_values=(0.0, 1.0),
            channels_first=False,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=None,
            device_type=device_type,
        )

        self._model.to(self.device)

        self._score_size = cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE
        self._score_size = self._score_size // cfg.ANCHOR.STRIDE
        self._score_size = self._score_size + cfg.TRACK.BASE_SIZE + 1

        self._generate_anchors()
        self._setup_window()
        self._reset()

        assert self.postprocessing_defences is None

    @property
    def input_shape(self) -> Tuple:
        #  N, F, H, W, C
        return (None, None, None, None, 3)

    @property
    def device(self) -> torch.device:
        return self._device

    def _generate_anchors(self):
        anchors = Anchors(cfg.ANCHOR.STRIDE, cfg.ANCHOR.RATIOS, cfg.ANCHOR.SCALES)
        anchors_np = anchors.anchors

        x1, y1, x2, y2 = (
            anchors_np[:, 0],
            anchors_np[:, 1],
            anchors_np[:, 2],
            anchors_np[:, 3],
        )
        anchors_np = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        num_anchors = anchors_np.shape[0]

        score_size = self._score_size
        total_stride = anchors.stride
        ori = -(score_size // 2) * total_stride
        xx, yy = np.meshgrid(
            [ori + total_stride * dx for dx in range(score_size)],
            [ori + total_stride * dy for dy in range(score_size)],
        )
        xx = np.tile(xx.flatten(), (num_anchors, 1)).flatten()
        yy = np.tile(yy.flatten(), (num_anchors, 1)).flatten()

        anchors_np = np.tile(anchors_np, score_size * score_size).reshape((-1, 4))
        anchors_np[:, 0] = xx.astype(np.float32)
        anchors_np[:, 1] = yy.astype(np.float32)
        self._anchors = torch.from_numpy(anchors_np).to(self.device)

    def _setup_window(self):
        num_anchors = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self._score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), num_anchors)
        self._window = torch.from_numpy(window).to(self.device)

    def _reset(self):
        self._trk_size = None
        self._trk_center_pos = None
        self._trk_pad_values = None

    def _rescale(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_values[1] == 255.0:
            return x
        elif self.clip_values[1] == 1.0:
            return x * 255.0
        else:
            raise ValueError(self.clip_values)

    def _convert_bbox(self, delta: torch.Tensor) -> torch.Tensor:
        delta = delta.permute(1, 2, 3, 0).view(4, -1).contiguous()

        anchors = self._anchors
        assert delta.size(1) == anchors.size(0), (delta.size(), anchors.size())

        delta[0, :] = delta[0, :] * anchors[:, 2] + anchors[:, 0]
        delta[1, :] = delta[1, :] * anchors[:, 3] + anchors[:, 1]
        delta[2, :] = torch.exp(delta[2, :]) * anchors[:, 2]
        delta[3, :] = torch.exp(delta[3, :]) * anchors[:, 3]
        return delta

    def _init_sequence(self, init_frame: torch.Tensor, gt_bbox: torch.Tensor):
        self._trk_size = bbox2corner(gt_bbox)[2:]
        self._trk_center_pos = bbox2center(gt_bbox)[:2]
        self._trk_pad_values = init_frame.mean(dim=(0, 1))

        s_z = compute_context_size(
            bbox_size=self._trk_size, context_amount=cfg.TRACK.CONTEXT_AMOUNT
        )

        z = crop_window(
            x=init_frame,
            center_pos=self._trk_center_pos.long().tolist(),  # type: ignore
            scale_size=s_z.long().item(),  # type: ignore
            output_size=cfg.TRACK.EXEMPLAR_SIZE,
            pad_values=self._trk_pad_values,
        )
        z = self._rescale(z)
        self._model.template(z)

    def _track_frame(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert frame.size(2) == 3
        assert isinstance(self._trk_size, torch.Tensor)

        s_z = compute_context_size(
            bbox_size=self._trk_size, context_amount=cfg.TRACK.CONTEXT_AMOUNT
        )
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        x = crop_window(
            x=frame,
            center_pos=self._trk_center_pos.long().tolist(),  # type: ignore
            scale_size=s_x.long().item(),  # type: ignore
            output_size=cfg.TRACK.INSTANCE_SIZE,
            pad_values=self._trk_pad_values,
        )
        x = self._rescale(x)
        outputs = self._model.track(x)

        score = outputs["cls"]  # shape: (N, 2k, H', W')
        score = score.permute(1, 2, 3, 0).contiguous()
        score = score.view(2, -1).permute(1, 0)  # shape: (N * k * H' * W', 2)
        score = F.softmax(score, dim=1)
        score = score[:, 1]

        pred_bbox = self._convert_bbox(outputs["loc"])

        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        prev_size_scaled = self._trk_size * scale_z

        prev_ctx_size = _ctx_size(prev_size_scaled[0], prev_size_scaled[1])
        curr_ctx_size = _ctx_size(pred_bbox[2, :], pred_bbox[3, :])

        prev_ratio = self._trk_size[0] / self._trk_size[1]
        curr_ratio = pred_bbox[2, :] / pred_bbox[3, :]

        scale_change = _expansion_rate(curr_ctx_size / prev_ctx_size)
        ratio_change = _expansion_rate(prev_ratio / curr_ratio)
        assert scale_change.size() == ratio_change.size() == score.size()

        # Apply scale+ratio penalty
        d = (scale_change * ratio_change) - 1
        p = torch.exp(-cfg.TRACK.PENALTY_K * d)
        pscore = p * score

        # Apply window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE)
        pscore = pscore + self._window * cfg.TRACK.WINDOW_INFLUENCE

        best_idx = pscore.argmax()

        bbox = pred_bbox[:, best_idx] / scale_z

        cx = bbox[0] + self._trk_center_pos[0]
        cy = bbox[1] + self._trk_center_pos[1]

        lr = p[best_idx] * score[best_idx] * cfg.TRACK.LR
        w = (lr * bbox[2]) + ((1 - lr) * self._trk_size[0])
        h = (lr * bbox[3]) + ((1 - lr) * self._trk_size[1])

        cx, cy, w, h = _clip_center(cx, cy, w, h, frame.shape[:2])

        # Update state
        self._trk_size = torch.stack([w, h])
        self._trk_center_pos = torch.stack([cx, cy])

        center = torch.stack([cx, cy, w, h])
        bbox = center2bbox(center)

        return bbox, pscore[best_idx]

    def forward(
        self, x: torch.Tensor, gt_bboxes: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        N, F, H, W, C = x.size()

        assert C == 3
        assert self.clip_values[0] <= x.min().item()
        assert x.max().item() <= self.clip_values[1]
        assert gt_bboxes.size() == (N, 4)

        output = list()
        for seq_idx in range(N):
            xi = x[seq_idx]
            gt_bbox_i = gt_bboxes[seq_idx]
            self._init_sequence(xi[0], gt_bbox_i)

            boxes, scores = list(), list()
            for frm_idx in range(F):
                box, score = self._track_frame(xi[frm_idx])
                boxes.append(box)
                scores.append(score)
            output.append(dict(boxes=torch.stack(boxes), scores=torch.stack(scores)))
            self._reset()

        return output

    def predict(
        self, x: np.ndarray, batch_size: int = 1, **kwargs
    ) -> List[Dict[str, np.ndarray]]:
        assert x.shape[0] == batch_size

        self._model.eval()

        y_init = kwargs.pop("y_init")
        if isinstance(x, np.ndarray):
            input_is_np = True
            x = torch.from_numpy(x).to(self.device)
            y = torch.from_numpy(y_init).to(self.device)
        elif isinstance(x, torch.Tensor):
            input_is_np = False
            x = x.to(self.device)
            y = y_init.to(self.device)

        x, y = self._apply_preprocessing(x, y, fit=False, no_grad=False)
        outputs = self.forward(x, gt_bboxes=y)

        if input_is_np:
            outputs = [
                {k: v.detach().cpu().numpy() for k, v in seq.items()} for seq in outputs
            ]
        return outputs

    def get_activations(
        self,
        x: np.ndarray,
        layer: Union[int, str],
        batch_size: int,
        framework: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError

    def loss_gradient(
        self, x: np.ndarray, y: List[Dict[str, np.ndarray]], **kwargs
    ) -> np.ndarray:
        assert isinstance(x, np.ndarray), type(x)
        N, F, H, W, C = x.shape
        assert C == 3

        grads: List[np.ndarray] = list()
        loss_fn = torch.nn.L1Loss(reduction="sum")

        for i in range(N):
            self._model.zero_grad()

            x_i = torch.from_numpy(x[[i]]).to(self.device).requires_grad_()
            y_i = np.expand_dims(y[i]["boxes"], axis=0)
            y_i = torch.from_numpy(y_i).to(self.device)

            preds = self.predict(x_i, y_init=y_i[:, 0])
            loss = loss_fn(preds[0]["boxes"], y_i[0].float())
            loss.backward()

            grads.append(x_i.grad[0].detach().cpu().numpy())

        output = np.stack(grads, axis=0)
        assert output.shape == x.shape, (output.shape, x.shape)
        return output

    def native_label_is_pytorch_format(self) -> bool:
        return True


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> ARTObjectTracker:

    cfg_path = model_kwargs.get("cfg_path")
    if cfg_path is not None:
        cfg_path = resolve_path(cfg_path)
        cfg.merge_from_file(cfg_path)
        logger.info("Merged cfg from %s", cfg_path)

    # Weights path is configured through cfg
    assert weights_path is None, weights_path

    return ARTObjectTracker(**wrapper_kwargs)
