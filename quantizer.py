# the vq-vae code is adapted from https://github.com/zalandoresearch/pytorch-vq-vae
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
    ):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim
        )
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings
        ).type_as(self._embedding.weight)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized_flat = torch.matmul(encodings, self._embedding.weight)
        quantized = quantized_flat.view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

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


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay,
        epsilon=1e-5,
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim
        )
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(
            torch.Tensor(num_embeddings, self._embedding_dim)
        )
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=inputs.device,
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized_flat = torch.matmul(encodings, self._embedding.weight)
        quantized = quantized_flat.view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

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


class LatticeQuantizer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        commitment_cost,
        sparsity_cost,
        initialize_embedding_b,
    ):
        super(LatticeQuantizer, self).__init__()

        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = commitment_cost
        self.gamma = sparsity_cost

        self.B = 1 / ((self.K ** (1 / self.D)) - 1)

        self.embedding = nn.Embedding(1, self.D)
        if initialize_embedding_b:
            self.embedding.weight.data.uniform_(-self.B, self.B)
        else:
            self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, latents: torch.Tensor):
        latents = latents.permute(
            0, 2, 3, 1
        ).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

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
            + self.beta * commitment_loss
            + self.gamma * size_loss
        )

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        # convert quantized from BHWC -> BCHW
        return {
            "vq_loss": lq_loss,
            "quantized": quantized_latents.permute(0, 3, 1, 2).contiguous(),
            "quantized_flat": quantized_latents_flat,
        }
