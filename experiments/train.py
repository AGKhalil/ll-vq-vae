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
    FashionMNISTDataModule,
    CelebADataModule,
)
from vector_quantizer import (
    VectorQuantizer,
    VectorQuantizerEMA,
    LatticeQuantizer,
)
from callbacks import GenerateCallback
import pytorch_lightning as pl


datasets = {
    "FashionMNIST": FashionMNISTDataModule,
    "CELEB-A": CelebADataModule,
    "FFHQ-1024": FFHQ1024DataModule,
}
quantizers = {
    "vq": VectorQuantizer,
    "vq-ema": VectorQuantizerEMA,
    "lattice": LatticeQuantizer,
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

    data = datasets[cfg.dataset.name](
        **cfg.dataset.dataset,
        batch_size=cfg.dataset.trainer.batch_size,
    )
    data.prepare_data()
    data.setup()
    quantizer = quantizers[cfg.quantizer.name](**cfg.quantizer.args)
    model = Model(
        **cfg.optimizer,
        **cfg.dataset.model,
        embedding_dim=cfg.quantizer.args.embedding_dim,
        quantizer=quantizer,
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
                count_uniques_bool=cfg.quantizer.count_uniques_bool,
                batch_size=cfg.dataset.trainer.batch_size,
                every_n_epochs=2,
            ),
        ],
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
