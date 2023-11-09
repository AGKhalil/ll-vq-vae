# Code from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/datamodules.html
import os
import deeplake
import pytorch_lightning as pl
from torchvision.datasets import (
    FashionMNIST,
    CelebA,
)
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


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


class CelebADataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.ds = deeplake.load(data_dir)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds.tensors["images"][idx].numpy()
        image = self.transform(image)
        return image


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
        if not os.path.exists(os.path.join(f"./{self.data_dir}", "celeba")):
            deeplake.deepcopy(
                "hub://activeloop/celeb-a-train",
                os.path.join(f"./{self.data_dir}", "celeba", "train"),
                tensors=["images"],
            )
            deeplake.deepcopy(
                "hub://activeloop/celeb-a-val",
                os.path.join(f"./{self.data_dir}", "celeba", "val"),
                tensors=["images"],
            )
            deeplake.deepcopy(
                "hub://activeloop/celeb-a-test",
                os.path.join(f"./{self.data_dir}", "celeba", "test"),
                tensors=["images"],
            )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = CelebADataset(
                os.path.join(f"./{self.data_dir}", "celeba", "train"),
                transform=self.transform,
            )
            self.val_dataset = CelebADataset(
                os.path.join(f"./{self.data_dir}", "celeba", "val"),
                transform=self.transform,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = CelebADataset(
                os.path.join(f"./{self.data_dir}", "celeba", "test"),
                transform=self.transform,
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
        if not os.path.exists(f"./{self.data_dir}"):
            deeplake.deepcopy(
                "hub://activeloop/ffhq",
                f"./{self.data_dir}",
                tensors=["images_1024/image", "images_1024/face_landmarks"],
            )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full = FFHQ1024Dataset(self.data_dir, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(
                full, [60000, 10000]
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
