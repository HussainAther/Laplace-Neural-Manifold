# **Laplace Neural Manifold**

A Python library for manifold learning with uncertainty estimation using the **Laplace approximation**. This project combines dimensionality reduction, neural networks, and Bayesian inference to create an interpretable and uncertainty-aware representation of high-dimensional data.

## **Overview**
The Laplace Neural Manifold provides:
- Dimensionality reduction via neural networks.
- Uncertainty quantification using the Laplace approximation.
- Visualization tools for understanding learned latent spaces.

This library is ideal for researchers and developers looking to incorporate uncertainty into manifold learning tasks, with applications ranging from data visualization to semi-supervised learning.

---

## **Key Features**
- **Manifold Learning**: Leverage autoencoders to map high-dimensional data into a low-dimensional latent space.
- **Uncertainty Estimation**: Apply the Laplace approximation to capture posterior distributions over the latent space.
- **Visualization Tools**: Visualize latent spaces with uncertainty overlays, using PCA, t-SNE, or UMAP.
- **Dataset Support**: Supports synthetic datasets (e.g., Swiss roll, S-curve) and real-world datasets like MNIST.

---

## **Getting Started**

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/HussainAther/Laplace-Neural-Manifold.git
   cd Laplace-Neural-Manifold
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Example Usage**

#### **Step 1: Train an Autoencoder**
Use the provided model to map high-dimensional data to a latent space:
```python
from src.model import Autoencoder
from src.utils import generate_swiss_roll
import torch

# Generate data
data = generate_swiss_roll()
input_dim = data.shape[1]
latent_dim = 2

# Initialize autoencoder
model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    latent, reconstructed = model(data)
    loss = criterion(reconstructed, data)
    loss.backward()
    optimizer.step()
```

#### **Step 2: Apply the Laplace Approximation**
Estimate uncertainty around the latent space:
```python
from src.laplace_approximation import apply_laplace
from torch.utils.data import DataLoader, TensorDataset

# Create DataLoader
loader = DataLoader(TensorDataset(data), batch_size=32, shuffle=True)

# Apply Laplace approximation
laplace = apply_laplace(model, loader, criterion)
```

#### **Step 3: Visualize Latent Space with Uncertainty**
Visualize the learned latent space and uncertainty estimates:
```python
from src.visualize import visualize_latent_space

# Extract latent points
latent_points, _ = model.encoder(data)
uncertainties = torch.sqrt(torch.diag(laplace.posterior_variance()))

# Visualize latent space
visualize_latent_space(latent_points.detach().numpy(), uncertainties.detach().numpy())
```

---

## **Project Structure**
```
Laplace-Neural-Manifold/
├── README.md               # Project overview
├── requirements.txt        # Dependencies
├── setup.py                # Installation script
├── src/                    # Core library
│   ├── model.py            # Autoencoder implementation
│   ├── laplace_approximation.py  # Laplace approximation logic
│   ├── visualize.py        # Visualization tools
│   ├── utils.py            # Utility functions (e.g., dataset generation)
├── notebooks/              # Example Jupyter notebooks
│   ├── example_latent_manifold.ipynb
├── examples/               # Script examples
│   ├── swiss_roll_example.py
├── tests/                  # Unit tests
│   ├── test_model.py
│   ├── test_laplace.py
├── data/                   # Placeholder for datasets
│   ├── README.md
└── docs/                   # Project documentation
```

---

## **Datasets**
### **Supported Datasets**
1. **Synthetic Data**:
   - Swiss Roll
   - S-Curve
2. **Real Data**:
   - MNIST (requires downloading)

### **Adding Your Dataset**
To use your own dataset, load it into a PyTorch `DataLoader` and pass it to the library functions.

---

## **Visualization**
The library provides tools for visualizing latent spaces:
- **Latent Space with Uncertainty**: Displays the 2D latent space with color-coded uncertainty estimates.
- **t-SNE and PCA Integration**: Easily visualize high-dimensional latent spaces.

Example output:

- **Latent Points** (color: ground truth labels, size: uncertainty)
  ![Example Visualization Placeholder](https://via.placeholder.com/800x400)

---

## **Contributing**
We welcome contributions! Here's how you can help:
1. Report bugs or suggest features through [issues](https://github.com/HussainAther/Laplace-Neural-Manifold/issues).
2. Submit pull requests for new features, bug fixes, or documentation improvements.

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Contact**
Created by **Syed Hussain Ather**. Feel free to reach out at **shussainather@gmail.com** or connect on [GitHub](https://github.com/HussainAther).
