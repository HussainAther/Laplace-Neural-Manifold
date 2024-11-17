import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder model for dimensionality reduction.
    Maps input data to a latent space and reconstructs it back to the original space.
    """
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            latent (torch.Tensor): Encoded latent representation.
            reconstructed (torch.Tensor): Reconstructed input data.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def encode(self, x):
        """
        Encodes the input into the latent space.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Encoded latent representation.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decodes the latent space representation back to input space.

        Args:
            z (torch.Tensor): Latent representation.

        Returns:
            torch.Tensor: Reconstructed input data.
        """
        return self.decoder(z)

