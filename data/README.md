### **`data/README.md`**

# Data Directory

This directory is intended to store datasets used in the **Laplace Neural Manifold** project. The project supports both synthetic and real-world datasets for training, testing, and visualization of latent spaces.

---

## **Contents**
This folder contains the following:
- **Synthetic Data**: Datasets generated programmatically (e.g., Swiss roll, S-curve).
- **Real-world Data**: Preprocessed or downloaded datasets like MNIST or CIFAR-10.

---

## **Supported Datasets**

### 1. **Synthetic Datasets**
- **Swiss Roll**: A 3D dataset representing a rolled sheet, ideal for testing manifold learning.
- **S-Curve**: A 3D dataset resembling an "S"-shaped curve.

These datasets are generated dynamically during the execution of the program, so no pre-downloading is required.

### 2. **Real-world Datasets**
If you plan to use real-world datasets, place them here. For example:
- **MNIST**: 28x28 grayscale images of digits (0-9).
- **CIFAR-10**: 32x32 RGB images from 10 classes.

> Note: Download scripts or links for real-world datasets may be provided in the project.

---

## **How to Use This Directory**

1. **Synthetic Datasets**: No action is needed. The project generates these dynamically.
2. **Real-world Datasets**:
   - Place any downloaded datasets in this directory.
   - Ensure the data format matches the expectations of your `DataLoader`.

---

## **Adding New Datasets**

If you want to add your own datasets:
1. Save the data files in this directory.
2. Update the project code to load these files (e.g., update `src/utils.py`).
3. Use the `DataLoader` class in PyTorch to feed the data into the model.

---

## **Example Structure**
Here’s an example of how this directory may look after adding some datasets:
```
data/
├── swiss_roll.npz            # Example pre-saved synthetic dataset (optional)
├── mnist/                    # MNIST dataset files
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
├── cifar10/                  # CIFAR-10 dataset files
│   ├── train/
│   ├── test/
├── custom_dataset.csv        # Example custom dataset file
```

---

## **Tips**
- For large datasets, consider linking this directory to an external storage location.
- Use the `src/utils.py` file to define dataset loaders for new datasets.

---

## **License**
Ensure you have the appropriate permissions and follow licensing requirements when using external datasets.
