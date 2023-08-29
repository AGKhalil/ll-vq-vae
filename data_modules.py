# Code from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/datamodules.html
import deeplake
import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.datasets import (
    MNIST,
    FashionMNIST,
    CIFAR10,
    CelebA,
)
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
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
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.full = MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.train_dataset, self.val_dataset = random_split(
                self.full, [55000, 5000]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
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
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )


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
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.full = FashionMNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.train_dataset, self.val_dataset = random_split(
                self.full, [55000, 5000]
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


class CIFAR10DataModule(pl.LightningDataModule):
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
                transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
            ]
        )

        self.num_classes = 10

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.full = CIFAR10(
                self.data_dir, train=True, transform=self.transform
            )
            self.train_dataset, self.val_dataset = random_split(
                self.full, [45000, 5000]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar10_test = CIFAR10(
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
            self.cifar10_test,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )


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
                transforms.Resize(64),
                transforms.ToTensor(),
            ]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
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


def ffhq_collate_fn(batch):
    return (
        torch.Tensor(
            np.array(
                [
                    np.transpose(image["images_1024/image"], (2, 0, 1))
                    for image in batch
                ]
            )
        ),
        None,
    )


class FFHQ1024Dataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.ds = deeplake.load(data_dir)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        landmarks = self.ds.tensors["images_1024/face_landmarks"][idx].numpy()
        image = self.ds.tensors["images_1024/image"][idx].numpy()
        image = self.transform(image)
        return image, landmarks


class FFHQ1024DataModule(pl.LightningDataModule):
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
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # deeplake.deepcopy(
        #     "hub://activeloop/ffhq",
        #     "./data/ffhq/ffhq-1024",
        #     tensors=["images_1024/image", "images_1024/face_landmarks"],
        # )
        # deeplake.deepcopy(
        #     "hub://activeloop/ffhq",
        #     "./data/ffhq/ffhq-128",
        #     tensors=["images_128/image"],
        # )
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.full = FFHQ1024Dataset(self.data_dir, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(
                self.full, [60000, 10000]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = FFHQ1024Dataset(
                self.data_dir, transform=self.transform
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
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )
