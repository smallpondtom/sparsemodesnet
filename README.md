# Sparse Mode Selection Neural Network (SparseModesNet)

SparseModesNet is a dimensionality reduction framework that employs linear encoding via POD modes and nonlinear NN decoding. The decoder leverages [LassoNet](https://github.com/lasso-net/lassonet), a method enforcing hierarchical sparsity through residual connections with linear skip layers, to simultaneously select informative POD modes and learn a nonlinear mapping that minimizes reconstruction error.

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
- `s`: Number of candidate POD modes
- `r`: Target number of active modes after sparsification
- `p`: Latent dimension of the nonlinear mapping

**Network Architecture:**
- `network_type`: Decoder type ('PiNetCCP', 'MLP', 'CNN', 'UNet', 'Rational')
- `poly_order`: Polynomial order (1=linear, 2=quadratic, 3=cubic)
- `hidden_units`: List of hidden layer sizes
- `num_polys`: Number of polynomial terms (for PiNet)

**Data Preprocessing:**
- `normalize_data`: Whether to normalize input data
- `center`: Center data around mean
- `whiten`: Apply ZCA whitening
- `normalize_type`: 'minmax' ([0,1]) or 'minmaxsym' ([-1,1])

**Sparse Training (LASSO phase):**
- `lam0`: Initial LASSO regularization parameter
- `lasso_epochs`: Number of LASSO training epochs
- `lasso_lr`: Learning rate for LASSO phase
- `epsilon`: Regularization increment factor
- `M`: Hierarchical constraint parameter

**Decoder Training:**
- `decoder_epochs`: Number of decoder training epochs
- `decoder_lr`: Learning rate for decoder
- `decoder_optimizer`: 'Adam' or 'SGD'
- `skip_sparse`: Skip sparse selection (train with fixed modes)

**Hardware:**
- `device`: 'cuda', 'mps', or 'cpu'
- `dtype`: 'float64' or 'float32'

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

- **PiNet (`pinet.py`)**: Polynomial network with combinatorial expansion
- **MLP (`mlp.py`)**: Standard multi-layer perceptron
- **CNN (`cnn.py`)**: Convolutional neural network
- **U-Net (`unet.py`)**: Encoder-decoder with skip connections

### Linear Algebra Utilities (`linalg/`)

- **POD (`pod.py`)**: Proper Orthogonal Decomposition
- **Least Squares (`lstsq.py`)**: Least squares solver

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

## Examples

See the `experiments/` directory for complete examples:

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{koike2026Sparse,
  title = {Sparse {{POD Mode Selection}} and {{Manifold Dimensionality Reduction}} with {{Neural Networks}}},
  author = {Koike, Tomoki and Mohan, Prakash and de Frahan, Marc T. Henry and Qian, Elizabeth and Bessac, Julie},
  year = 2026,
  month = may,
  publisher = {arXiv},
  doi = {10.48550/arXiv.2605.27756},
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, please contact [Tomoki Koike](mailto:tkoike45@gmail.com).