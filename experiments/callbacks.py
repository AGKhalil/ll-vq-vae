import wandb
import pytorch_lightning as pl

from utils import count_uniques_in_batch, reconstruct_images


class GenerateCallback(pl.Callback):
    def __init__(self, batch_size, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.batch_size = batch_size

    def reconstruct_images(self, trainer, pl_module, train=True):
        grid = reconstruct_images(
            dataloader=trainer.datamodule.test_dataloader, model=pl_module
        )
        caption_name = "train" if train else "val"
        images = wandb.Image(grid, caption=f"{caption_name} reconstructions")
        trainer.logger.experiment.log(
            {
                f"{caption_name}/reconstructions": images,
                "global_step": trainer.global_step,
            }
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.reconstruct_images(trainer, pl_module, train=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.reconstruct_images(trainer, pl_module, train=False)

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        if batch_idx % 100 == 0:
            uniques, total_shape = count_uniques_in_batch(
                model=pl_module,
                batch=batch,
            )
            trainer.logger.experiment.log(
                {
                    "batch_n_uniques": uniques.shape[0],
                    "batch_total_n": total_shape,
                }
            )
