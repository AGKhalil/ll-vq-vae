# Code from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/datamodules.html
import os
import deeplake
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


class FFHQ1024Dataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.ds = deeplake.load(data_dir)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds.tensors["images_1024/image"][idx].numpy()
        image = self.transform(image)
        return image, idx


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
                tensors=["images_1024/image"],
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
