# the vq-vae code is adapted from https://github.com/zalandoresearch/pytorch-vq-vae
import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super(Quantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        raise NotImplementedError()
