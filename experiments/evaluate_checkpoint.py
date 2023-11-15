import json
import os
import matplotlib.pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from utils import count_uniques, reconstruct_images
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
    config_name="eval",
)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)
    hydra_cfg = HydraConfig.get()
    quantizer_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)[
        "quantizer"
    ]
    dataset_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)[
        "dataset"
    ]
    checkpoint_path = os.path.join(
        "pretrained",
        dataset_choice,
        f"{quantizer_choice}.ckpt",
    )
    checkpoint = torch.load(checkpoint_path)

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
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    grid = reconstruct_images(dataloader=data.test_dataloader, model=model)
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.savefig(
        os.path.join(
            "reconstructions",
            dataset_choice,
            f"{quantizer_choice}.png",
        ),
        dpi=600,
    )

    if "dense" in quantizer_choice:
        print(
            "The dense lattice assigns a quantization point per embedding"
            " vector. So the codebook is too large to count."
        )
        return

    codebook_usage = count_uniques(
        model=model,
        dataloaders=[
            data.train_dataloader,
            data.val_dataloader,
        ],
    )

    json_path = os.path.join(
        "reconstructions",
        dataset_choice,
        f"{quantizer_choice}.json",
    )
    with open(json_path, "w") as json_file:
        json.dump(codebook_usage, json_file)


if __name__ == "__main__":
    main()
