from typing import Dict

import pytorch_lightning as pl
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR

from skelevision.config import cfg
from skelevision.models.model_builder import MTLModelBuilder


class LitMTLModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MTLModelBuilder()

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = SGD(
            trainable_params,
            lr=cfg.TRAIN.BASE_LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )

        lr_scheduler = ExponentialLR(optimizer, gamma=cfg.TRAIN.GAMMA)

        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def forward(  # type:ignore
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def training_step(  # type:ignore
        self, batch: Dict[str, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        output = self.model(batch)
        for k, v in output.items():
            log_kwargs = dict(on_step=True, on_epoch=True)
            if k == "total_loss":
                log_kwargs.update(prog_bar=True)
            self.log(f"train_{k}", v, **log_kwargs)  # type:ignore
        return output.pop("total_loss")

    def validation_step(  # type:ignore
        self, batch: Dict[str, torch.Tensor], batch_index: int
    ) -> torch.Tensor:
        output = self.model(batch)
        for k, v in output.items():
            log_kwargs = dict(on_step=True, on_epoch=True)
            if k == "total_loss":
                log_kwargs.update(prog_bar=True)
            self.log(f"valid_{k}", v, **log_kwargs)  # type:ignore
        return output.pop("total_loss")
