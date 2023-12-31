# model architecture code is adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
from typing import List
from torch import Tensor
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from vector_quantizer import Quantizer


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class Model(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: List[int] = None,
        in_channels: int = None,
        learning_rate: float = 1e-3,
        quantizer: Quantizer = None,
        scheduler_gamma: float = 0.0,
        weight_decay=0.0,
    ):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        self.scheduler_gamma = scheduler_gamma
        self.weight_decay = weight_decay
        self._quantizer = quantizer

        out_channels = in_channels

        modules = []
        hidden_dims = hidden_dims

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, in_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.LeakyReLU(),
            )
        )

        for _ in range(2):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )

        self._encoder = nn.Sequential(*modules)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    embedding_dim,
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.LeakyReLU(),
            )
        )

        for _ in range(2):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.LeakyReLU(),
            )
        )

        self._decoder = nn.Sequential(*modules)

    def forward(self, x):
        z = self._encoder(x)
        vq_output = self._quantizer(z)
        x_recon = self._decoder(
            vq_output["quantized"],
        )

        return {
            "data_recon": x_recon,
            **{key: value for key, value in vq_output.items() if "vq" in key},
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        recon_error = F.mse_loss(output["data_recon"], x)
        loss = recon_error + output["vq_loss"]

        log_dict = {}
        if "perplexity" in output:
            perplexity = output["perplexity"]
            log_dict["perplexity"] = perplexity
        log_dict["vq_loss"] = output["vq_loss"]
        log_dict["recon_error"] = recon_error
        log_dict["loss"] = loss

        self.log_dict(
            {f"train_{key}": train.item() for key, train in log_dict.items()},
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        recon_error = F.mse_loss(output["data_recon"], x)
        loss = recon_error + output["vq_loss"]

        log_dict = {}
        if "perplexity" in output:
            perplexity = output["perplexity"]
            log_dict["perplexity"] = perplexity
        log_dict["vq_loss"] = output["vq_loss"]
        log_dict["recon_error"] = recon_error
        log_dict["loss"] = loss

        self.log_dict(
            {f"val_{key}": val.item() for key, val in log_dict.items()},
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        recon_error = F.mse_loss(output["data_recon"], x)
        loss = recon_error + output["vq_loss"]

        log_dict = {}
        if "perplexity" in output:
            perplexity = output["perplexity"]
            log_dict["perplexity"] = perplexity
        log_dict["vq_loss"] = output["vq_loss"]
        log_dict["recon_error"] = recon_error
        log_dict["loss"] = loss

        self.log_dict(
            {f"test_{key}": test.item() for key, test in log_dict.items()},
            sync_dist=True,
        )

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        return pred
