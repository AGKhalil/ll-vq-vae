from typing import List, Tuple
import torch
import torchvision
from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import DataLoader


def count_uniques_in_batch(
    model: Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, int]:
    """
    Count the number of unique quantized embeddings (codewords) in a batch.

    Args:
        model (Module): The model.
        batch (Tuple[torch.Tensor, torch.Tensor]): The input batch.

    Returns:
        Tuple[torch.Tensor, int]: Unique values and total shape.

    """
    inputs, _ = batch
    pre_vq_output = model._encoder(inputs.to(model.device))
    vq_output = model._quantizer(pre_vq_output)
    quantized_flats = vq_output["quantized_flat"]
    return torch.unique(quantized_flats, dim=0), quantized_flats.shape[0]


def count_uniques(
    model: Module, dataloaders: List[DataLoader]
) -> Tuple[int, int]:
    """
    Count the number of unique quantized embeddings (codewords) in a list of dataloaders.

    Args:
        model (Module): The model.
        dataloaders (List[DataLoader]): The dataloaders.

    Returns:
        Tuple[int, int]: Number of unique values and total number of samples.

    """
    all_quantized_flats = []
    all_ns = []
    with torch.no_grad():
        for dataloader in dataloaders:
            for _, batch in tqdm(
                enumerate(dataloader()),
                total=len(dataloader()),
            ):
                uniques, total_shape = count_uniques_in_batch(model, batch)
                all_quantized_flats.append(uniques)
                all_ns.append(total_shape)
        all_quantized_flats_stack = torch.concat(all_quantized_flats)
        n_uniques = torch.unique(all_quantized_flats_stack, dim=0).shape[0]
        total_n = sum(all_ns)

    return n_uniques, total_n


def reconstruct_images(
    dataloader: DataLoader, model: Module, sample_size: int = 8
) -> torch.Tensor:
    """
    Reconstruct images from a dataloader.

    Args:
        dataloader (DataLoader): The dataloader.
        model (Module): The model.
        sample_size (int, optional): Number of samples to reconstruct. Defaults to 8.

    Returns:
        torch.Tensor: Reconstructed images in a grid.

    """
    with torch.no_grad():
        inputs, _ = next(iter(dataloader()))
        sample = inputs[:sample_size, :, :, :]
        pre_vq_output = model._encoder(sample.to(model.device))
        vq_output = model._quantizer(pre_vq_output)
        reconst_imgs = model._decoder(vq_output["quantized"])

        imgs = torch.stack(
            [sample.to(model.device), reconst_imgs], dim=1
        ).flatten(0, 1)
        grid = torchvision.utils.make_grid(
            imgs,
            nrow=4,
        )
        return grid
