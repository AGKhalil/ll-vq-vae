import os
import hydra
from omegaconf import DictConfig
from dotenv import dotenv_values
import omegaconf
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import LearningRateMonitor


from model import Model
from data_modules import (
    FFHQ1024DataModule,
    MNISTDataModule,
    FashionMNISTDataModule,
    CIFAR10DataModule,
    CelebADataModule,
)
from callbacks import GenerateCallback

import pytorch_lightning as pl


datasets = {
    "MNIST": MNISTDataModule,
    "FashionMNIST": FashionMNISTDataModule,
    "CIFAR10": CIFAR10DataModule,
    "CELEB-A": CelebADataModule,
    "FFHQ-1024": FFHQ1024DataModule,
}


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.general.seed)

    if "WANDB_SECRET_PATH" in os.environ:
        os.environ["WANDB_API_KEY"] = dotenv_values(
            os.environ["WANDB_SECRET_PATH"]
        )["WANDB_API_KEY"]
    wandb_logger = pl.loggers.WandbLogger(
        entity=cfg.general.entity,
        project=cfg.general.project,
        log_model=True,
    )
    if rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(
            omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
    wandb_logger.experiment.log_code(".")

    data = datasets[cfg.dataset.dataset.name](
        data_dir=cfg.dataset.dataset.data_dir,
        batch_size=cfg.dataset.trainer.batch_size,
        num_workers=cfg.dataset.dataset.num_workers,
    )
    data.setup()
    model = Model(
        **cfg.optimizer,
        **cfg.model.args,
        in_channels=cfg.dataset.dataset.in_channels,
    )
    wandb_logger.watch(model)
    trainer = pl.Trainer(
        max_epochs=cfg.dataset.trainer.max_epochs,
        logger=wandb_logger,
        default_root_dir="checkpoints/",
        accelerator="gpu",
        devices=[cfg.general.device],
        callbacks=[
            LearningRateMonitor(),
            GenerateCallback(
                count_uniques_bool=cfg.model.count_uniques_bool,
                batch_size=cfg.dataset.trainer.batch_size,
                every_n_epochs=2,
            ),
        ],
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
