import pytest
import torch
from src.model import Autoencoder

@pytest.fixture
def sample_data():
    # Create sample input data
    return torch.rand((10, 3))  # 10 samples, 3 features each

@pytest.fixture
def autoencoder():
    # Initialize an autoencoder with 3 input dimensions and 2 latent dimensions
    return Autoencoder(input_dim=3, latent_dim=2)

def test_autoencoder_forward(autoencoder, sample_data):
    # Test the forward pass
    latent, reconstructed = autoencoder(sample_data)

    # Ensure output shapes are correct
    assert latent.shape == (10, 2), "Latent dimension mismatch"
    assert reconstructed.shape == (10, 3), "Reconstructed dimension mismatch"

def test_autoencoder_encode(autoencoder, sample_data):
    # Test the encoder function
    latent = autoencoder.encode(sample_data)

    # Ensure latent space output shape is correct
    assert latent.shape == (10, 2), "Encoded dimension mismatch"

def test_autoencoder_decode(autoencoder):
    # Test the decoder function
    latent = torch.rand((10, 2))  # Random latent points
    reconstructed = autoencoder.decode(latent)

    # Ensure reconstructed shape matches input space
    assert reconstructed.shape == (10, 3), "Decoded dimension mismatch"

