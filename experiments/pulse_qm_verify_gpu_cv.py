#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable
from sklearn.model_selection import TimeSeriesSplit

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.pulse import generate_advecting_pulse

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.5 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]

def quadratic_mapping(x):
    """
    Vectorized computation of unique Kronecker product x ⊗ x.
    Only computes upper triangular part to avoid redundancy.
    
    Args:
        x: torch.Tensor of shape (batch_size, n) or (n,)
        
    Returns:
        torch.Tensor of shape (batch_size, n*(n+1)//2) or (n*(n+1)//2,)
    """
    if not isinstance(x, torch.Tensor):
        dim = x.ndim
    else:
        dim = x.dim()
    
    if dim == 1:
        n = x.size(0)
        # Create indices for upper triangular part
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        # Compute products
        result = x[i_indices] * x[j_indices]
        return result
    else:
        batch_size, n = x.shape
        # Create indices for upper triangular part  
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        # Compute products for all batches
        result = x[:, i_indices] * x[:, j_indices]
        return result   

class QuadraticManifold(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, gamma: float, W: torch.Tensor = None):
        super(QuadraticManifold, self).__init__()
        
        self.register_buffer('U_r', pod_basis)  # (d, r): store as buffer
        self.d, self.r = pod_basis.shape
        if W is None:
            self.weight_mat = nn.Parameter(
                torch.randn(self.r * (self.r + 1) // 2, self.d) * 0.01)
        else:
            self.weight_mat = nn.Parameter(W, requires_grad=True)  # (r*(r+1)//2, d)
        self.gamma = gamma  # Regularization parameter
        
    def forward(self, z_batch):
        # Reconstruct the linear part via projection
        x_hat_lin = z_batch @ self.U_r.T     # (batch, d)
        # Apply the quadratic mapping
        z_quad = quadratic_mapping(z_batch)  # (batch, r*(r+1)//2)
        x_hat_nn = z_quad @ self.weight_mat  # (batch, d)
        # Reconstruct x_hat
        x_hat = x_hat_lin + x_hat_nn
        return x_hat

def train_qm_single(model: QuadraticManifold, num_epochs: int, lr: float,
                   device: str, z_train: np.ndarray, x_train: np.ndarray,
                   z_val: np.ndarray = None, x_val: np.ndarray = None, 
                   verbose: bool = True):
    """
    Train a single QuadraticManifold model
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=model.gamma)
    mse_loss = nn.MSELoss(reduction='mean')
    
    train_loss_history = []
    val_loss_history = []
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000)
    
    z_train_tensor = torch.tensor(z_train, dtype=torch.float32)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    
    if z_val is not None and x_val is not None:
        z_val_tensor = torch.tensor(z_val, dtype=torch.float32)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        z_train_batch = z_train_tensor.to(device)
        x_train_batch = x_train_tensor.to(device)
        
        optimizer.zero_grad()
        x_hat_batch = model(z_train_batch)
        reconstruction_loss = mse_loss(x_hat_batch, x_train_batch)
        loss = reconstruction_loss
        loss.backward()
        optimizer.step()

        # Calculate training error
        with torch.no_grad():
            x_hat_batch = model(z_train_batch)
            train_rel_error = torch.norm(x_hat_batch - x_train_batch) / torch.norm(x_train_batch)
            train_loss_history.append(train_rel_error.item())
        
        # Calculate validation error if validation data provided
        val_rel_error = None
        if z_val is not None and x_val is not None:
            model.eval()
            with torch.no_grad():
                z_val_batch = z_val_tensor.to(device)
                x_val_batch = x_val_tensor.to(device)
                x_val_hat = model(z_val_batch)
                val_rel_error = torch.norm(x_val_hat - x_val_batch) / torch.norm(x_val_batch)
                val_loss_history.append(val_rel_error.item())
                
                # Save best model
                if val_rel_error.item() < best_val_loss:
                    best_val_loss = val_rel_error.item()
                    best_model_state = deepcopy(model.state_dict())
        
        # Use validation loss for scheduler if available, otherwise training loss
        scheduler_loss = val_rel_error.item() if val_rel_error is not None else train_rel_error.item()
        lr_scheduler.step(scheduler_loss)

        # Print progress
        if verbose and ((epoch % 1000 == 0) or (epoch == 1)):
            lr_current = optimizer.param_groups[0]['lr']
            print_str = f"  Epoch {epoch:<6d} | lr={lr_current:.4e} | Train Error={train_rel_error.item():.6e}"
            if val_rel_error is not None:
                print_str += f" | Val Error={val_rel_error.item():.6e}"
            print(print_str)
    
    # Load best model if validation was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_loss_history, val_loss_history, best_val_loss if val_rel_error is not None else train_loss_history[-1]


def train_qm_cv(pod_basis: torch.Tensor, gamma: float, W_init: torch.Tensor,
                num_epochs: int, lr: float, device: str, 
                z_data: np.ndarray, x_data: np.ndarray, 
                DataSplitter: Callable, n_splits: int = 5):
    """
    Train QuadraticManifold using scikit-learn's TimeSeriesSplit cross-validation
    
    Uses expanding window approach where training set grows with each fold.
    
    Args:
        pod_basis: POD basis matrix
        gamma: Regularization parameter
        W_init: Initial weight matrix
        num_epochs: Number of training epochs per fold
        lr: Learning rate
        device: Device to train on
        z_data: Reduced coordinates (n_samples, r)
        x_data: Target data (n_samples, d)
        n_splits: Number of CV splits
    
    Returns:
        cv_scores: List of validation scores for each fold
        best_model: Best performing model
        fold_histories: Training/validation histories for each fold
    """
    
    print(f"\n{'='*60}")
    print(f"Training with {n_splits}-Fold Time Series Cross-Validation")
    print(f"{'='*60}")
    
    # Initialize cross-validation splitter
    tscv = DataSplitter(n_splits=n_splits)
    
    cv_scores = []
    fold_histories = []
    best_score = float('inf')
    best_model = None
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(z_data)):
        print(f"\nFold {fold + 1}/{n_splits}")
        print(f"Train indices: {train_idx[0]} to {train_idx[-1]} (size: {len(train_idx)})")
        print(f"Val indices: {val_idx[0]} to {val_idx[-1]} (size: {len(val_idx)})")
        
        # Split data
        z_train_fold = z_data[train_idx]
        x_train_fold = x_data[train_idx]
        z_val_fold = z_data[val_idx]
        x_val_fold = x_data[val_idx]
        
        # Initialize new model for this fold
        model = QuadraticManifold(
            pod_basis, 
            gamma, 
            W_init.clone() + torch.randn_like(W_init) * 0.01
        )
        
        # Train model
        train_history, val_history, val_score = train_qm_single(
            model, num_epochs, lr, device,
            z_train_fold, x_train_fold,
            z_val_fold, x_val_fold,
            verbose=True
        )
        
        cv_scores.append(val_score)
        fold_histories.append({
            'train_history': train_history,
            'val_history': val_history,
            'train_idx': train_idx,
            'val_idx': val_idx
        })
        
        # Save best model
        if val_score < best_score:
            best_score = val_score
            best_model = deepcopy(model)
        
        print(f"Fold {fold + 1} validation error: {val_score:.6e}")
    
    print(f"\n{'='*60}")
    print("Cross-Validation Results")
    print(f"{'='*60}")
    print(f"CV Scores: {[f'{score:.6e}' for score in cv_scores]}")
    print(f"Mean CV Score: {np.mean(cv_scores):.6e}")
    print(f"Std CV Score: {np.std(cv_scores):.6e}")
    print(f"Best CV Score: {np.min(cv_scores):.6e}")
    
    return cv_scores, best_model, fold_histories


def plot_cv_results(fold_histories, cv_scores):
    """Plot cross-validation results"""
    n_folds = len(fold_histories)
    
    # Plot training/validation curves for each fold
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation curves
    ax1 = axes[0, 0]
    for i, fold_data in enumerate(fold_histories):
        epochs = range(1, len(fold_data['train_history']) + 1)
        ax1.semilogy(epochs, fold_data['train_history'], 
                    label=f'Fold {i+1} Train', alpha=0.7)
        ax1.semilogy(epochs, fold_data['val_history'], 
                    label=f'Fold {i+1} Val', alpha=0.7, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Relative Error')
    ax1.set_title('Training/Validation Curves by Fold')
    ax1.legend()
    ax1.grid(True)
    
    # CV scores
    ax2 = axes[0, 1]
    ax2.bar(range(1, n_folds + 1), cv_scores)
    ax2.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                label=f'Mean: {np.mean(cv_scores):.2e}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Validation Error')
    ax2.set_title('Cross-Validation Scores')
    ax2.legend()
    ax2.grid(True)
    
    # Data split visualization
    ax3 = axes[1, 0]
    for i, fold_data in enumerate(fold_histories):
        train_idx = fold_data['train_idx']
        val_idx = fold_data['val_idx']
        ax3.barh(i, len(train_idx), left=train_idx[0], 
                alpha=0.7, label='Train' if i == 0 else "")
        ax3.barh(i, len(val_idx), left=val_idx[0], 
                alpha=0.7, label='Val' if i == 0 else "")
    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('Fold')
    ax3.set_title('Time Series Split Visualization')
    ax3.legend()
    ax3.grid(True)
    
    # Final validation curves only
    ax4 = axes[1, 1]
    for i, fold_data in enumerate(fold_histories):
        epochs = range(1, len(fold_data['val_history']) + 1)
        ax4.semilogy(epochs, fold_data['val_history'], 
                    label=f'Fold {i+1}', alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Error')
    ax4.set_title('Validation Curves Comparison')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


#%%
if __name__ == "__main__":
    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Using device:", device)
    
    # Number of modes
    r_max = 15
    
    # number of grids
    n_grids = 2**10
    
    # Sanity check flag (plotting)
    sanity_check = True

    # ---------- Advecting Pulse ----------
    X_pulse, xspan_p, tspan_p = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=8.0,
        final_time=0.15,
        n_time_samples=1000,
        n_space_samples=n_grids
    )
    d_p, n_p = X_pulse.shape
    s_p = min(d_p, n_p)
    s_p = 100
    
    ## Create 3D surface plot for Advecting Pulse (sanity check)
    if sanity_check:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X_mesh, T_mesh = np.meshgrid(xspan_p, tspan_p)
        Z_mesh = X_pulse.T  # Transpose to match meshgrid dimensions
        surf = ax.plot_surface(
            X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('Advecting Gaussian Pulse')
        plt.colorbar(surf, shrink=0.5, aspect=5)
        # plt.savefig('../figures/pulse_data.png', dpi=300)
        plt.show()
        plt.close(fig)
        
    #%% Greedy Quadratic Manifold approach
    from QM.quadmani import quadmani_greedy, lift_quadratic, linear_reduce
    V, W, shift_value, I_qm = quadmani_greedy(
        X_pulse, r_max, s_p, 1e-6, np.array([], dtype=int))
    reduced_points = linear_reduce(V, X_pulse, shift_value)
    reconstructed = lift_quadratic(V, W, shift_value, reduced_points)
    rel_rec_error = np.linalg.norm(reconstructed - X_pulse) / np.linalg.norm(X_pulse)
    print('Relative reconstruction error: ', rel_rec_error)
    print("Quadratic manifold indices I_qm:", I_qm)
    shift_value = np.array(shift_value)[:, np.newaxis]

    #%% Train Neural Network Quadratic Manifold with Cross-Validation
    print("\n" + "="*60)
    print("Training Neural Network Quadratic Manifold with Time Series CV")
    print("="*60)
    
    # Use the linear basis V from greedy approach
    pod_basis = torch.tensor(V, dtype=torch.float32)
    
    # Training parameters
    gamma = 1e-6  # Regularization parameter
    num_epochs = 10000  # Reduced for CV
    lr = 1e-4
    n_splits = 5  # Number of CV folds
    
    print(f"Model parameters:")
    print(f"  Input dimension (d): {V.shape[0]}")
    print(f"  Reduced dimension (r): {V.shape[1]}")
    print(f"  Quadratic features: {V.shape[1] * (V.shape[1] + 1) // 2}")
    print(f"  Regularization gamma: {gamma}")
    print(f"  CV folds: {n_splits}")
    print(f"  Epochs per fold: {num_epochs}")
    print(f"  Learning rate: {lr}")
    
    # Prepare training data
    z_data = reduced_points.T              # (n_samples, r)
    x_data = (X_pulse - shift_value).T     # (n_samples, d)
    
    print(f"\nData shapes:")
    print(f"  z_data: {z_data.shape}")
    print(f"  x_data: {x_data.shape}")
    print(f"  Total time samples: {z_data.shape[0]}")
    
    # Initial weight matrix
    W_init = torch.tensor(W, dtype=torch.float32).T + torch.randn(W.T.shape, dtype=torch.float32) * 0.01
    
    # Train with cross-validation
    # NOTE: you can replace BlockingTimeSeriesSplit with TimeSeriesSplit 
    # for non-blocking CV
    cv_scores, best_model, fold_histories = train_qm_cv(
        pod_basis, gamma, W_init, num_epochs, lr, device, 
        z_data, x_data, BlockingTimeSeriesSplit, n_splits
    )
    
    #%% Evaluate best model on full dataset
    print("\n" + "="*60)
    print("Evaluating Best Model on Full Dataset")
    print("="*60)
    
    best_model.eval()
    with torch.no_grad():
        z_test = torch.tensor(z_data, dtype=torch.float32).to(device)
        x_reconstructed = best_model(z_test).cpu().numpy().astype(np.float64)  
        x_reconstructed += shift_value.T  # Add the shift back
    
    # Compute reconstruction error
    rel_error_cv = np.linalg.norm(x_reconstructed.T - X_pulse) / np.linalg.norm(X_pulse)
    print(f"Relative reconstruction error (CV Best Model): {rel_error_cv:.6e}")
    print(f"Relative reconstruction error (Greedy QM): {rel_rec_error:.6e}")
    print(f"Improvement ratio: {rel_rec_error / rel_error_cv:.3e}x")
    
    #%% Plot cross-validation results
    plot_cv_results(fold_histories, cv_scores)