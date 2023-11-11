# Code from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/datamodules.html
import os
import pytorch_lightning as pl
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class FashionMNISTDataModule(pl.LightningDataModule):
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
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.0,)),
            ]
        )

        self.num_classes = 10

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.exists(f"./{self.data_dir}/FashionMNIST"):
            FashionMNIST(self.data_dir, train=True, download=True)
            FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full = FashionMNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.train_dataset, self.val_dataset = random_split(
                full, [55000, 5000]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.fashionmnist_test = FashionMNIST(
                self.data_dir, train=False, transform=self.transform
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
            self.fashionmnist_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )
