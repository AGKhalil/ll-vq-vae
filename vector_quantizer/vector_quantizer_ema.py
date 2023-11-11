# the vq-vae code is adapted from https://github.com/zalandoresearch/pytorch-vq-vae
import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantizer.quantizer import Quantizer


class VectorQuantizerEMA(Quantizer):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super(VectorQuantizerEMA, self).__init__(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
        )
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(
            torch.Tensor(num_embeddings, self.embedding_dim)
        )
        self.ema_w.data.normal_()

        self.decay = decay
        self.epsilon = epsilon

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
            encoding_indices.shape[0],
            self.num_embeddings,
            device=inputs.device,
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized_flat = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized_flat.view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (
                1 - self.decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(
                self.ema_w * self.decay + (1 - self.decay) * dw
            )

            self.embedding.weight = nn.Parameter(
                self.ema_w / self.ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
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
