# HipMRI VQ-VAE — Model architecture modules
# Author: Aayushi Arvind Dhande (s4831001)
# Description: Implements the Residual encoder-decoder, Vector Quantizer (EMA),
#              and the full VQ-VAE model for 2D HipMRI slice reconstruction.

"""Model architecture components for the HipMRI VQ-VAE project.

This module defines all building blocks required for Vector-Quantized Variational
Autoencoder (VQ-VAE) training and inference:
- ResidualBlock and ResidualStack for non-linear feature refinement.
- Encoder and Decoder networks for latent compression and reconstruction.
- VectorQuantizer for discrete latent embedding via exponential moving average (EMA).
- VQVAE wrapper combining all components into an end-to-end trainable model.

References:
    Oord, A. van den, Vinyals, O., & Kavukcuoglu, K. (2017).
    "Neural Discrete Representation Learning" (VQ-VAE).
"""

from math import log2
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block used in Encoder and Decoder stacks.

    Each block performs two 3×3 convolutions with BatchNorm and ReLU, followed
    by a residual skip connection scaled by a learnable parameter `alpha`.

    Args:
        in_channels (int): Number of input channels.
        res_channels (int): Number of channels in the intermediate residual path.
    """
    def __init__(self, in_channels: int, res_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        # Learnable scaling factor for the residual output.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with scaled residual connection."""
        return self.layers(x) * self.alpha + x

class ResidualStack(nn.Module):
    """Stack of multiple ResidualBlocks.

    Args:
        in_channels (int): Number of input channels.
        res_channels (int): Number of channels in residual paths.
        nb_layers (int): Number of residual layers to stack.
    """
    def __init__(self, in_channels: int, res_channels: int, nb_layers: int) -> None:
        super(ResidualStack, self).__init__()
        self.stack = nn.Sequential(*[ResidualBlock(in_channels, res_channels) for _ in range(nb_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all residual layers sequentially."""
        return self.stack(x)

class Encoder(nn.Module):
    """Encoder network that downsamples spatial features to a latent map.

    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale).
        hidden_channels (int): Base number of channels in the feature hierarchy.
        res_channels (int): Channels for residual sub-blocks.
        nb_res_layers (int): Number of residual layers after downsampling.
        downscale_factor (int): Overall downscale factor (must be power of 2).
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, res_channels: int, nb_res_layers: int, downscale_factor: int) -> None:
        super(Encoder, self).__init__()
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2

        # Downsampling path: halving spatial size at each step.
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, hidden_channels

         # Final convolution and residual refinement.
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to a latent feature map."""
        return self.layers(x)

class Decoder(nn.Module):
     """Decoder network that reconstructs an image from quantized latents.

    Args:
        in_channels (int): Channels in the latent input.
        hidden_channels (int): Intermediate channels for upsampling layers.
        out_channels (int): Number of channels in the reconstructed output.
        res_channels (int): Channels for residual sub-blocks.
        nb_res_layers (int): Number of residual layers after upsampling.
        upscale_factor (int): Overall upsampling factor (must be power of 2).
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, res_channels: int, nb_res_layers: int, upscale_factor: int) -> None:
        super(Decoder, self).__init__()
        assert log2(upscale_factor) % 1 == 0, "Upscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        # Upsampling path: doubles spatial size at each step.
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, out_channels
            
        # Final convolution and normalization.    
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image tensor from the latent representation."""
        return self.layers(x)

class VectorQuantizer(nn.Module):
    """Vector quantization layer implementing EMA codebook updates.

    Args:
        in_channels (int): Input channel size from encoder.
        embed_dim (int): Dimension of embedding vectors.
        nb_entries (int): Number of embedding entries in the codebook.

    Notes:
        This version uses exponential moving averages (EMA) to update the
        embedding vectors, stabilizing training as in VQ-VAE v2.
    """
    
    def __init__(self, in_channels: int, embed_dim: int, nb_entries: int) -> None:
        super(VectorQuantizer, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)
        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5

         # Initialize codebook and EMA tracking buffers.
        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize encoder output using nearest embeddings.

        Args:
            x (torch.Tensor): Input latent tensor [B, C, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Quantized tensor [B, C, H, W].
                - Commitment/codebook loss term.
                - Embedding index map [B, H, W].
        """
        # Project input to embedding dimension.
        x = self.conv_in(x).permute(0, 2, 3, 1)
        flatten = x.reshape(-1, self.dim)

        # Compute pairwise distances to all codebook entries.
        dist = (
            flatten.pow(2).sum(1, keepdim=True) 
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        # Nearest embedding lookup.
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        # EMA updates for codebook centroids.
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n)
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # Compute codebook (commitment) loss and apply straight-through estimator. 
        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()
        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
         """Map embedding indices to corresponding codebook vectors."""
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class VQVAE(nn.Module):
    """Complete VQ-VAE model combining Encoder, VectorQuantizer, and Decoder."""
    def __init__(self, in_channels: int, hidden_channels: int, res_channels: int, nb_res_layers: int, embed_dim: int, nb_entries: int, downscale_factor: int) -> None:
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, downscale_factor)
        self.code_layer = VectorQuantizer(hidden_channels, embed_dim, nb_entries)
        self.decoder = Decoder(embed_dim, hidden_channels, in_channels, res_channels, nb_res_layers, downscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full VQ-VAE pipeline.

        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Reconstructed output [B, C, H, W].
                - Codebook commitment loss scalar.
                - Embedding indices [B, H', W'].
        """
        
        encoded = self.encoder(x)
        quantized, diff, embed_ind = self.code_layer(encoded)
        decoded = self.decoder(quantized)
        return decoded, diff, embed_ind


