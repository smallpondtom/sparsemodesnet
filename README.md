# Sparse Mode Selection Neural Network (SparseModesNet)

A Python package for learning sparse, nonlinear reduced-order models from high-dimensional data using neural networks with polynomial feature mappings.

## Overview

SparseModesNet combines ideas from dimensionality reduction, sparse optimization, and deep learning to discover interpretable low-dimensional representations of complex dynamical systems. The package implements:

- **Polynomial Networks (Π-Nets)**: Neural networks with polynomial feature mappings (quadratic, cubic, etc.)
- **Sparse Mode Selection**: Automatic selection of important POD modes via iterative hard thresholding
- **Multiple Decoder Architectures**: MLP, CNN, U-Net, and rational function decoders
- **Flexible Training Pipeline**: Dense-to-sparse training with adaptive regularization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sparsemodesnet.git
cd sparsemodesnet

# Install in development mode
pip install -e .
```

## Quick Start

```python
import numpy as np
import sparsemodesnet as smn

# Load your data (features × samples)
X = np.load('your_data.npy')

# Configure the model
config = smn.SparseModesNetConfig(
    s=100,              # Number of POD modes
    r=15,               # Target number of active modes
    p=300,              # Latent dimension
    network_type='PiNetCCP',
    poly_order=2,       # Quadratic features
    hidden_units=[50, 600, 300],
    device='cuda',      # Use 'cuda', 'mps', or 'cpu'
    dtype='float64'     # 'float64' or 'float32'
)

# Train the model
model, selected_modes, omegas, history, reconstruction_error = smn.fit(X, config)

# Visualize mode selection evolution
smn.omega_evolve(omegas, selected_modes, config.s, save=True)
```

## Core Components

### Configuration (`config.py`)

`SparseModesNetConfig` is the main configuration class that controls all aspects of training:

**Dimensionality Parameters:**
- `s`: Number of POD modes to retain
- `r`: Target number of active modes after sparsification
- `p`: Latent dimension of the autoencoder

**Network Architecture:**
- `network_type`: Decoder type ('PiNetCCP', 'MLP', 'CNN', 'UNet', 'Rational')
- `poly_order`: Polynomial order (1=linear, 2=quadratic, 3=cubic)
- `hidden_units`: List of hidden layer sizes
- `num_polys`: Number of polynomial terms (for PiNet)

**Data Preprocessing:**
- `normalize_data`: Whether to normalize input data
- `center`: Center data around mean
- `whiten`: Apply ZCA whitening
- `normalize_type`: 'minmax', 'standard', or 'robust'

**Sparse Training (LASSO phase):**
- `lam0`: Initial LASSO regularization parameter
- `lasso_epochs`: Number of LASSO training epochs
- `lasso_lr`: Learning rate for LASSO phase
- `epsilon`: Thresholding parameter for mode selection
- `M`: Lipschitz constant bound

**Decoder Training:**
- `decoder_epochs`: Number of decoder training epochs
- `decoder_lr`: Learning rate for decoder
- `decoder_optimizer`: 'Adam' or 'SGD'
- `skip_sparse`: Skip sparse selection (train with fixed modes)

**Hardware:**
- `device`: 'cuda', 'mps', or 'cpu'
- `dtype`: 'float64' (default for CUDA) or 'float32'

### Training Pipeline (`fit.py`)

The main `fit()` function orchestrates the entire training process:

```python
model, I_nn, omegas, path_history, reconstruction_error = smn.fit(X, config)
```

**Returns:**
- `model`: Trained PyTorch model
- `I_nn`: Indices of selected modes
- `omegas`: Evolution of mode selection weights
- `path_history`: Training history
- `reconstruction_error`: Final reconstruction error

### Decoder Models (`decoder_models/`)

Multiple decoder architectures are available:

1. **PiNet (`pinet.py`)**: Polynomial network with combinatorial expansion
   - Efficient polynomial feature computation
   - Supports quadratic (Π²) and cubic (Π³) mappings
   
2. **MLP (`mlp.py`)**: Standard multi-layer perceptron
   - Flexible hidden layer configuration
   - Batch normalization and dropout options

3. **CNN (`cnn.py`)**: Convolutional neural network
   - For spatial/image data
   - Transpose convolutions for upsampling

4. **U-Net (`unet.py`)**: Encoder-decoder with skip connections
   - Preserves spatial information
   - Good for high-resolution reconstructions

5. **Rational (`rational.py`)**: Rational function approximation
   - Learns polynomial ratio representations

### Linear Algebra Utilities (`linalg/`)

- **POD (`pod.py`)**: Proper Orthogonal Decomposition
  - Randomized SVD for large datasets
  - Energy threshold-based truncation

- **ZCA Whitening (`zca.py`)**: Zero-phase component analysis
  - Decorrelates features
  - Preserves spatial structure

- **Least Squares (`lstsq.py`)**: Robust least squares solvers
  - Handles ill-conditioned problems
  - Regularization options

### Training Modules (`training/`)

- **Dense-to-Sparse (`dense2sparse.py`)**: 
  - Iterative hard thresholding algorithm
  - Automatic mode selection via ω weights
  - Proximal gradient descent

- **Train (`train.py`)**:
  - Standard training loop
  - Learning rate scheduling
  - Early stopping criteria

### Visualization (`viz/`)

- **omega_evolve.py**: Visualize mode selection evolution
  ```python
  smn.omega_evolve(
      omegas, 
      selected_modes, 
      num_modes=100,
      save=True,
      filename='omega_evolution.pdf'
  )
  ```

## Advanced Usage

### Custom Mode Selection

```python
# Use pre-selected modes (e.g., from greedy algorithm)
config = smn.SparseModesNetConfig(
    skip_sparse=True,
    I_nn=range(15),  # Use first 15 modes
    # ... other parameters
)
```

### Mixed Precision Training

```python
# Use Float32 for faster CUDA training
config = smn.SparseModesNetConfig(
    device='cuda',
    dtype='float32',  # Faster but less precise
    # ... other parameters
)
```

### Data Preprocessing

```python
from sparsemodesnet.preprocess import preprocess_data

# Preprocess your data
X_processed, mean, std, U_pod = preprocess_data(
    X,
    s=100,
    normalize=True,
    center=True,
    normalize_type='minmax'
)
```

### Custom Datasets

```python
from sparsemodesnet.dataset import SparseModeDataset
from torch.utils.data import DataLoader

# Create custom dataset
dataset = SparseModeDataset(X, U_pod, mean, std)
loader = DataLoader(dataset, batch_size=200, shuffle=True)
```

## Architecture Details

### Polynomial Networks (Π-Nets)

Π-Nets use polynomial feature mappings to capture nonlinear relationships:

- **Π²-Net**: Quadratic features → `n(n+1)/2` features
- **Π³-Net**: Cubic features → `n(n+1)(n+2)/6` features

The polynomial expansion is computed efficiently using vectorized operations:

```python
# Quadratic mapping: [x₁, x₂, ...] → [x₁², x₁x₂, x₂², ...]
φ₂(x) = [xᵢxⱼ for i ≤ j]

# Cubic mapping: [x₁, x₂, ...] → [x₁³, x₁²x₂, x₁x₂², x₂³, ...]
φ₃(x) = [xᵢxⱼxₖ for i ≤ j ≤ k]
```

### Sparse Mode Selection Algorithm

The iterative hard thresholding algorithm:

1. Initialize ω weights to ones
2. For each iteration:
   - Compute gradient of reconstruction loss
   - Update ω via proximal gradient
   - Hard threshold: keep top-k modes with largest |ω|
   - Update model parameters
3. Stop when mode selection stabilizes

## Examples

See the `experiments/` directory for complete examples:

- `kse.py`: Kuramoto-Sivashinsky equation
- Additional examples coming soon

## Performance Tips

1. **CUDA Acceleration**: Use `device='cuda'` with Float64 for best accuracy
2. **Batch Size**: Larger batches (200-500) for stable training
3. **Learning Rate**: Start with 1e-2 for decoder, 1e-3 for LASSO
4. **Mode Selection**: Start with larger `s` (100-200) and smaller `r` (10-20)
5. **Polynomial Order**: Use Π²-Net first, try Π³-Net if needed

## Citation

If you use this code in your research, please cite:

```bibtex
@article{koike2026SparseModesNet,
  title={Sparse POD Mode Selection and Manifold Dimensionality Reduction with Neural Networks},
  author={Tomoki Koike, Prakash Mohan, Marc T. de Henry Frahan, Elizabeth Qian, and Julia Bessac},
  journal={Journal of Tentative},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, please contact [Tomoki Koike](mailto:tkoike45@gmail.com).