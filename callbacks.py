import torch
import torchvision
from tqdm import tqdm
import wandb
import pytorch_lightning as pl
import numpy as np


class GenerateCallback(pl.Callback):
    def __init__(self, batch_size, name, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.batch_size = batch_size
        self.name = name

    def reconstruct_images(self, trainer, pl_module, train=True):
        with torch.no_grad():
            inputs, _ = next(iter(trainer.datamodule.val_dataloader()))
            pre_vq_output = pl_module._encoder(inputs.to(pl_module.device))
            vq_output = pl_module._vq_vae(pre_vq_output)
            reconst_imgs = pl_module._decoder(vq_output["quantized"])

            # Plot and add to logger
            imgs = torch.stack(
                [inputs.to(pl_module.device), reconst_imgs], dim=1
            ).flatten(0, 1)
            grid = torchvision.utils.make_grid(
                imgs,
                nrow=4,
            )
            caption_name = "train" if train else "val"
            images = wandb.Image(
                grid, caption=f"{caption_name} reconstructions"
            )
            trainer.logger.experiment.log(
                {
                    f"{caption_name}/reconstructions": images,
                    "global_step": trainer.global_step,
                }
            )

    def count_uniques_in_batch(self, pl_module, data):
        inputs, _ = data
        pre_vq_output = pl_module._encoder(inputs.to(pl_module.device))
        vq_output = pl_module._vq_vae(pre_vq_output)
        quantized_flats = vq_output["quantized_flat"].cpu()
        return torch.unique(quantized_flats, dim=0), quantized_flats.shape[0]

    def count_uniques(self, trainer, pl_module):
        all_quantized_flats = []
        all_ns = []
        with torch.no_grad():
            for _, data in tqdm(
                enumerate(trainer.datamodule.train_dataloader()),
                total=len(trainer.datamodule.train_dataloader()),
            ):
                uniques, total_shape = self.count_uniques_in_batch(
                    pl_module, data
                )
                all_quantized_flats.append(uniques)
                all_ns.append(total_shape)
            all_quantized_flats_stack = torch.concat(all_quantized_flats)
            training_dataset_n_uniques = torch.unique(
                all_quantized_flats_stack, dim=0
            ).shape[0]
            training_dataset_total_n = sum(all_ns)

        trainer.logger.experiment.log(
            {
                "training_dataset_n_uniques": training_dataset_n_uniques,
                "training_dataset_total_n": training_dataset_total_n,
            }
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.reconstruct_images(trainer, pl_module, train=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.reconstruct_images(trainer, pl_module, train=False)

    def on_train_end(self, trainer, pl_module) -> None:
        self.count_uniques(trainer, pl_module)

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        if batch_idx % 100 == 0:
            uniques, total_shape = self.count_uniques_in_batch(pl_module, batch)
            trainer.logger.experiment.log(
                {
                    "batch_n_uniques": uniques.shape[0],
                    "batch_total_n": total_shape,
                }
            )
