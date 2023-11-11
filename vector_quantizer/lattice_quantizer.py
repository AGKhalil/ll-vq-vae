# the vq-vae code is adapted from https://github.com/zalandoresearch/pytorch-vq-vae
import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantizer.quantizer import Quantizer


class LatticeQuantizer(Quantizer):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        sparsity_cost: float = 1.0,
        initialize_embedding_b: bool = True,
    ):
        super(LatticeQuantizer, self).__init__(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
        )
        self.sparsity_cost = sparsity_cost

        self.B = 1 / ((self.num_embeddings ** (1 / self.embedding_dim)) - 1)

        self.embedding = nn.Embedding(1, self.embedding_dim)
        if initialize_embedding_b:
            self.embedding.weight.data.uniform_(-self.B, self.B)
        else:
            self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, latents: torch.Tensor):
        latents = latents.permute(
            0, 2, 3, 1
        ).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        flat_latents = latents.view(-1, self.embedding_dim)  # [BHW x D]

        # Babai estimate
        babai_estimate = torch.round(
            torch.mul(flat_latents, 1 / self.embedding.weight)
        )

        # Quantize the latents
        quantized_latents_flat = torch.mul(
            self.embedding.weight, babai_estimate
        )
        quantized_latents = quantized_latents_flat.view(latents.shape)

        # Compute the LQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        size_loss = -torch.sum(torch.abs(self.embedding.weight))

        lq_loss = (
            embedding_loss
            + self.commitment_cost * commitment_loss
            + self.sparsity_cost * size_loss
        )

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # convert quantized from BHWC -> BCHW
        return {
            "vq_loss": lq_loss,
            "quantized": quantized_latents.permute(0, 3, 1, 2).contiguous(),
            "quantized_flat": quantized_latents_flat,
        }
