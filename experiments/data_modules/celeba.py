# Code from https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/datamodules.html
import os
import torch
import deeplake
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.default_collate(batch)


class CelebADataset(Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        self.ds = deeplake.load(data_dir)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds.tensors["images"][idx].numpy()
        box = self.ds.tensors["boxes"][idx].numpy()[0]

        if box[2] == box[3] == 0:
            return None

        image = self.transform(
            image[
                int(box[1]) : int(box[1] + box[3]),
                int(box[0]) : int(box[0] + box[2]),
                :,
            ]
        )
        return image, idx


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
                transforms.ToPILImage(),
                transforms.Resize(178),
                transforms.CenterCrop(178),
                transforms.ToTensor(),
            ]
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.exists(os.path.join(f"./{self.data_dir}")):
            deeplake.deepcopy(
                "hub://activeloop/celeb-a-train",
                os.path.join(f"./{self.data_dir}", "train"),
                tensors=[
                    "images",
                    "boxes",
                ],
            )
            deeplake.deepcopy(
                "hub://activeloop/celeb-a-val",
                os.path.join(f"./{self.data_dir}", "val"),
                tensors=[
                    "images",
                    "boxes",
                ],
            )
            deeplake.deepcopy(
                "hub://activeloop/celeb-a-test",
                os.path.join(f"./{self.data_dir}", "test"),
                tensors=[
                    "images",
                    "boxes",
                ],
            )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = CelebADataset(
                os.path.join(f"./{self.data_dir}", "train"),
                transform=self.transform,
            )
            self.val_dataset = CelebADataset(
                os.path.join(f"./{self.data_dir}", "val"),
                transform=self.transform,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = CelebADataset(
                os.path.join(f"./{self.data_dir}", "test"),
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=custom_collate_fn,
        )
