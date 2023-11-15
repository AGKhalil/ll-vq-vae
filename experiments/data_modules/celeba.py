# Code from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/datamodules.html
import os
from torchvision.datasets import CelebA
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.ToTensor(),
            ]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.exists(f"./{self.data_dir}"):
            CelebA(self.data_dir, split="train", transform=self.transform)
            CelebA(self.data_dir, split="valid", transform=self.transform)
            CelebA(self.data_dir, split="test", transform=self.transform)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = CelebA(
                self.data_dir, split="train", transform=self.transform
            )
            self.val_dataset = CelebA(
                self.data_dir, split="valid", transform=self.transform
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.celeb_test = CelebA(
                self.data_dir, split="test", transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.celeb_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )
