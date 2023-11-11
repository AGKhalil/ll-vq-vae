# the vq-vae code is adapted from https://github.com/zalandoresearch/pytorch-vq-vae
import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantizer.quantizer import Quantizer


class VectorQuantizer(Quantizer):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
    ):
        super(VectorQuantizer, self).__init__(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
        )
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings
        ).type_as(self.embedding.weight)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized_flat = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized_flat.view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        # convert quantized from BHWC -> BCHW
        return {
            "vq_loss": loss,
            "quantized": quantized.permute(0, 3, 1, 2).contiguous(),
            "vq_perplexity": perplexity,
            "encodings": encodings,
            "quantized_flat": quantized_flat,
        }
