import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import Autoencoder
from src.laplace_approximation import apply_laplace, extract_uncertainties

@pytest.fixture
def sample_data():
    # Create sample input data and targets
    inputs = torch.rand((100, 3))  # 100 samples, 3 features
    targets = torch.rand((100, 3))  # 100 samples, 3 targets
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=10)

@pytest.fixture
def trained_autoencoder():
    # Initialize and train an autoencoder
    autoencoder = Autoencoder(input_dim=3, latent_dim=2)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Create synthetic data for training
    inputs = torch.rand((100, 3))
    for _ in range(10):  # Train for 10 epochs
        optimizer.zero_grad()
        latent, reconstructed = autoencoder(inputs)
        loss = criterion(reconstructed, inputs)
        loss.backward()
        optimizer.step()

    return autoencoder

def test_apply_laplace(trained_autoencoder, sample_data):
    # Test Laplace approximation application
    laplace = apply_laplace(trained_autoencoder, sample_data)

    # Ensure Laplace object is returned
    assert laplace is not None, "Laplace approximation failed"
    assert hasattr(laplace, "posterior_variance"), "Missing posterior variance attribute"

def test_extract_uncertainties(trained_autoencoder, sample_data):
    # Test uncertainty extraction
    laplace = apply_laplace(trained_autoencoder, sample_data)
    means, uncertainties = extract_uncertainties(laplace, sample_data)

    # Ensure shapes match input batch sizes
    assert means.shape[0] == 100, "Mean predictions size mismatch"
    assert uncertainties.shape[0] == 100, "Uncertainty size mismatch"

