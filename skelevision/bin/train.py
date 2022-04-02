from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_info

from skelevision.config import cfg
from skelevision.data.datamodule import MTLDataModule
from skelevision.models.lit_model import LitMTLModel


def cli() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("config_path", type=Path)
    return parser


def main():
    args = cli().parse_args()

    config_path = args.config_path
    model_dir = config_path.parent
    cfg.merge_from_file(config_path)

    rank_zero_info("Loaded config from %s", config_path)
    rank_zero_info(cfg)

    seed_everything(cfg.TRAIN.SEED, workers=True)

    tblogger = TensorBoardLogger(str(model_dir), name="", version="")
    output_dir = Path(tblogger.log_dir).resolve()
    if len(list(output_dir.glob("*.ckpt"))) > 0:
        raise FileExistsError(next(output_dir.glob("*.ckpt")))

    lit_model = LitMTLModel()
    datamodule = MTLDataModule()

    lr_monitor = LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best_valid_total_loss",
        monitor="valid_total_loss",
        mode="min",
    )

    trainer = Trainer(
        logger=tblogger,
        callbacks=[lr_monitor, checkpoint_callback],
        strategy=DDPPlugin(find_unused_parameters=False),
        **cfg.TRAINER.KWARGS
    )

    trainer.fit(lit_model, datamodule)
    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
