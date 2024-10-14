import torch
import torch.nn as nn
from einops import rearrange
from model.base import MBase

class ViTAutoencoder(MBase):
    def __init__(self, embed_dim=128, num_patches=64, sequence_length=10, num_heads=4, transformer_layers=6, mlp_dim=256):
        """
        Args:
            embed_dim (int): Dimension of patch embeddings.
            num_patches (int): Number of patches per image.
            sequence_length (int): Length of the input sequence.
            num_heads (int): Number of attention heads in the transformer.
            transformer_layers (int): Number of transformer encoder layers.
            mlp_dim (int): Dimension of the MLP block inside the transformer.
        """
        super(ViTAutoencoder, self).__init__()
        
        # Encoder: Transformer Encoder Layers
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim),
            num_layers=transformer_layers
        )
        
        # Decoder: Another stack of Transformer Encoder Layers for reconstruction
        self.decoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim),
            num_layers=transformer_layers
        )

        # Projection layer to reduce sequence length (latent space representation)
        self.latent_projection = nn.Linear(embed_dim * num_patches, embed_dim * num_patches // 2)

        # Reverse projection layer to expand latent representation back to original sequence length
        self.reverse_projection = nn.Linear(embed_dim * num_patches // 2, embed_dim * num_patches)
    
    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor of shape (batch_size, sequence_length, num_patches, embed_dim)
                        e.g., (batch_size, 10, 64, 128)
        Returns:
            Reconstructed tensor of the same shape as input.
        """
        # Reshape input to (batch_size, sequence_length, num_patches * embed_dim)
        batch_size, sequence_length, num_patches, embed_dim = x.shape
        x = rearrange(x, 'b t p d -> b t (p d)')
        
        # Encode sequence: Pass through transformer encoder layers
        encoded_sequence = self.encoder_layers(x)  # Shape: (batch_size, sequence_length, num_patches * embed_dim)
        
        # Project to latent space
        latent = self.latent_projection(encoded_sequence)  # Shape: (batch_size, sequence_length, num_patches * embed_dim // 2)
        
        # Reverse projection to expand back to original patch size
        expanded_latent = self.reverse_projection(latent)  # Shape: (batch_size, sequence_length, num_patches * embed_dim)
        
        # Decode sequence: Pass through transformer decoder layers
        reconstructed_sequence = self.decoder_layers(expanded_latent)  # Shape: (batch_size, sequence_length, num_patches * embed_dim)
        
        # Reshape back to (batch_size, sequence_length, num_patches, embed_dim)
        reconstructed_sequence = rearrange(reconstructed_sequence, 'b t (p d) -> b t p d', p=num_patches, d=embed_dim)
        
        return reconstructed_sequence
