import pytorch_lightning as pl
from torch.utils.data import DataLoader

from skelevision.config import cfg
from skelevision.data.augmentation import get_training_aug_image
from skelevision.data.augmentation import get_training_aug_template
from skelevision.data.dataset import MTLDataset


class MTLDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        dataset = MTLDataset(
            dataset_split="train",
            image_aug=get_training_aug_image(),
            template_aug=get_training_aug_template(),
            template_strategy=cfg.TRAIN.TEMPLATE_STRATEGY,
        )
        return DataLoader(
            dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=5,
        )

    def val_dataloader(self):
        dataset = MTLDataset(dataset_split="valid", template_strategy="previous")
        return DataLoader(
            dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=5,
        )
