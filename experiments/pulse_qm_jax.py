#%%
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from flax.training import train_state
from typing import Any
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.pulse import generate_advecting_pulse

# Configure JAX for Apple GPU (JAX-Metal)
print("Available JAX devices:", jax.devices())
if len(jax.devices()) > 0 and 'metal' in str(jax.devices()[0]).lower():
    print("✓ JAX-Metal detected and will be used for Apple GPU acceleration")
else:
    print("ℹ JAX-Metal not detected, using CPU")

#========================= JAX Quadratic Mapping Function =========================#
def quadratic_mapping_jax(x):
    """
    JAX version of vectorized computation of unique Kronecker product x ⊗ x.
    
    Args:
        x: jnp.ndarray of shape (batch_size, n) or (n,)
        
    Returns:
        jnp.ndarray of shape (batch_size, n*(n+1)//2) or (n*(n+1)//2,)
    """
    if x.ndim == 1:
        n = x.shape[0]
        i_indices, j_indices = jnp.tril_indices(n)
        result = x[i_indices] * x[j_indices]
        return result
    else:
        _, n = x.shape
        i_indices, j_indices = jnp.tril_indices(n)
        result = x[:, i_indices] * x[:, j_indices]
        return result

#%% #==================== JAX Quadratic Manifold Class ============================#
class QuadraticManifold(nn.Module):
    """JAX/Flax implementation of QuadraticManifold using Flax Linen."""
    
    pod_basis: jnp.ndarray  # (d, r) - POD basis
    gamma: float            # Regularization parameter
    init_weights: Any = None  # Optional initial weights
    
    def setup(self):
        """Initialize the weight matrix parameter."""
        self.d, self.r = self.pod_basis.shape
        expected_shape = (self.r * (self.r + 1) // 2, self.d)
        
        if self.init_weights is not None:
            # Use provided initial weights
            if self.init_weights.shape == expected_shape:
                init_fn = lambda rng, shape, dtype: self.init_weights.astype(dtype)
            elif self.init_weights.shape == (expected_shape[1], expected_shape[0]):
                init_fn = lambda rng, shape, dtype: self.init_weights.T.astype(dtype)
            else:
                raise ValueError(f"Initial weights shape {self.init_weights.shape} "
                               f"doesn't match expected {expected_shape}")
        else:
            # Zero initialization
            init_fn = nn.initializers.zeros
        
        self.weight_mat = self.param(
            'weight_mat',
            init_fn,
            expected_shape,
            jnp.float64
        )
    
    def __call__(self, z_batch):
        """Forward pass through the quadratic manifold."""
        # Ensure double precision
        z_batch = z_batch.astype(jnp.float64)
        
        # Linear reconstruction
        x_hat_lin = z_batch @ self.pod_basis.T     # (batch, d)
        
        # Quadratic reconstruction  
        z_quad = quadratic_mapping_jax(z_batch)    # (batch, r*(r+1)//2)
        x_hat_quad = z_quad @ self.weight_mat      # (batch, d)
        
        # Total reconstruction
        x_hat = x_hat_lin + x_hat_quad
        return x_hat

#%% #===================== Loss Function and Training State ===================#
def compute_loss(params, apply_fn, z_batch, x_target, gamma):
    """Compute the loss function with L2 regularization."""
    x_pred = apply_fn(params, z_batch)
    reconstruction_loss = jnp.mean((x_pred - x_target) ** 2)
    regularization = gamma * jnp.sum(params['params']['weight_mat'] ** 2)
    return reconstruction_loss + regularization

def create_train_state(model, params, learning_rate):
    """Create the training state with optimizer."""
    # Use SGD with momentum (equivalent to PyTorch's SGD)
    optimizer = optax.sgd(learning_rate=learning_rate, momentum=0.99)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

@jax.jit
def train_step(state, z_batch, x_target, gamma):
    """Single training step (JIT compiled for speed)."""
    def loss_fn(params):
        return compute_loss(params, state.apply_fn, z_batch, x_target, gamma)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit  
def compute_relative_error(state, z_batch, x_target):
    """Compute relative reconstruction error (JIT compiled)."""
    x_pred = state.apply_fn(state.params, z_batch)
    return jnp.linalg.norm(x_pred - x_target) / jnp.linalg.norm(x_target)

#%% #======================== Training Function =====================#
def train_quadratic_manifold(pod_basis, z_train, x_target, gamma, 
                           init_weights=None, learning_rate=1e-4, 
                           num_epochs=10000, print_every=100):
    """
    Train the JAX QuadraticManifold model.
    
    Args:
        pod_basis: POD basis matrix (d, r)
        z_train: Training reduced coordinates (n_samples, r)
        x_target: Target reconstructions (n_samples, d)
        gamma: Regularization parameter
        init_weights: Optional initial weights for the quadratic terms
        learning_rate: Learning rate for SGD
        num_epochs: Number of training epochs
        print_every: Print frequency for training progress
        
    Returns:
        Trained model and final training state
    """
    
    # Convert to JAX arrays with double precision
    pod_basis = jnp.array(pod_basis, dtype=jnp.float64)
    z_train = jnp.array(z_train, dtype=jnp.float64)
    x_target = jnp.array(x_target, dtype=jnp.float64)
    
    if init_weights is not None:
        init_weights = jnp.array(init_weights, dtype=jnp.float64)
    
    # Initialize model
    model = QuadraticManifold(
        pod_basis=pod_basis, 
        gamma=gamma, 
        init_weights=init_weights
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    sample_input = z_train[:1]  # Use first sample for initialization
    params = model.init(key, sample_input)
    
    # Create training state
    state = create_train_state(model, params, learning_rate)
    
    # Training loop
    print("Starting JAX training...")
    losses = []
    
    for epoch in range(num_epochs):
        # Training step
        state, loss = train_step(state, z_train, x_target, gamma)
        losses.append(float(loss))
        
        # Print progress
        if epoch % print_every == 0:
            rel_error = compute_relative_error(state, z_train, x_target)
            current_lr = state.opt_state.hyperparams['learning_rate'] if hasattr(state.opt_state, 'hyperparams') else learning_rate
            print(f"Epoch {epoch:4d}: "
                  f"LR = {current_lr:.4e}, "
                  f"Loss = {loss:.4e}, "
                  f"Rel Error = {rel_error:.4e}")
    
    return model, state, losses

#%% #======================== Evaluation Function =====================#
def evaluate_model(model, state, z_test, shift_value, X_original):
    """Evaluate the trained model and compute reconstruction error."""
    
    # Get predictions
    x_reconstructed = model.apply(state.params, z_test)
    x_final = np.array(x_reconstructed) + shift_value.T
    
    # Compute relative error
    rel_error = np.linalg.norm(x_final.T - X_original) / np.linalg.norm(X_original)
    
    return x_final, rel_error

#%% #======================== Example Usage =====================#
def run_jax_training_example(V, W, shift_value, reduced_points, X, gamma):
    """
    Example function showing how to use the JAX implementation.
    This replaces the "Training the Neural Network" section.
    """
    
    print(f"\n{'='*60}")
    print("JAX NEURAL NETWORK TRAINING")
    print(f"{'='*60}")
    
    # Prepare data (matching the original PyTorch format)
    z_train = reduced_points.T.astype(np.float64)  # (n_samples, r)
    x_target = (X - shift_value).T.astype(np.float64)  # (n_samples, d)
    
    print(f"z_train shape: {z_train.shape}")
    print(f"x_target shape: {x_target.shape}")
    print(f"Using gamma: {gamma}")
    
    # Train with analytical initialization (like the PyTorch version)
    model, trained_state, losses = train_quadratic_manifold(
        pod_basis=V,
        z_train=z_train,
        x_target=x_target,
        gamma=gamma,
        init_weights=None,  # Transpose to match expected shape
        learning_rate=1e-4,
        num_epochs=100000,
        print_every=100
    )
    
    # Evaluate final model
    x_final_trained, rel_error_trained = evaluate_model(
        model, trained_state, z_train, shift_value, X
    )
    
    print(f"\nFinal JAX trained NN error: {rel_error_trained:.2e}")
    
    # Compare with original greedy QM solution
    from QM.quadmani import lift_quadratic
    reconstructed_greedy = lift_quadratic(V, W, shift_value, reduced_points)
    rel_error_greedy = np.linalg.norm(reconstructed_greedy - X) / np.linalg.norm(X)
    print(f"Original greedy QM error: {rel_error_greedy:.2e}")
    
    # Check weight matrix changes
    final_weights = np.array(trained_state.params['params']['weight_mat'])
    initial_weights = W.T
    weight_diff = np.max(np.abs(final_weights - initial_weights))
    print(f"Weight matrix change: {weight_diff:.2e}")
    
    if rel_error_trained < rel_error_greedy:
        print("✓ JAX training improved the model!")
    elif np.abs(rel_error_trained - rel_error_greedy) < 1e-10:
        print("✓ JAX training converged to analytical solution.")
    else:
        print("✗ JAX training performed worse than analytical solution.")
    
    return model, trained_state, losses

# Example of how to call this in your main script:
"""
# Replace the PyTorch training section with:
model, trained_state, losses = run_jax_training_example(
    V, W, shift_value, reduced_points, X_ks, gamma
)
"""


#%%
if __name__ == "__main__":
    # Force CPU for deterministic results
    device = 'cpu'
    print("Using device:", device)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
     

#%% #===================== Main Experiment with Pulse Data ====================#
    # Parameters
    r_max = 15
    n_grids = 2**10
    sanity_check = False  # Disable plotting for now
    
    print("\n" + "="*60)
    print("GENERATING DATA")
    print("="*60)
    
    # Generate advecting pulse data
    X_pulse, xspan_p, tspan_p = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.15,
        n_time_samples=1000,
        n_space_samples=n_grids
    )
    
    print(f"X_pulse shape: {X_pulse.shape}")
    print(f"X_pulse dtype: {X_pulse.dtype}")
    
    # Ensure data is double precision
    X_pulse = X_pulse.astype(np.float64)
    
    d_p, n_p = X_pulse.shape
    s_p = min(d_p, n_p)
    s_p = 100
    
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
    plt.show()
    plt.close(fig)
    
    
#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    print("="*60)
    
    # Get greedy QM solution
    from QM.quadmani import quadmani_greedy, lift_quadratic, linear_reduce
    
    gamma = 1e-6  # Regularization parameter
    
    V, W, shift_value, I_qm = quadmani_greedy(
        X_pulse, r_max, s_p, gamma, np.array([], dtype=int))
    
    # Ensure double precision
    V = V.astype(np.float64)
    W = W.astype(np.float64)
    shift_value = np.array(shift_value, dtype=np.float64)[:, np.newaxis]
    
    print(f"V shape: {V.shape}, dtype: {V.dtype}")
    print(f"W shape: {W.shape}, dtype: {W.dtype}")
    print(f"shift_value shape: {shift_value.shape}, dtype: {shift_value.dtype}")
    
    # Get reduced coordinates
    reduced_points = linear_reduce(V, X_pulse, shift_value)
    reduced_points = reduced_points.astype(np.float64)
    print(f"reduced_points shape: {reduced_points.shape}, dtype: {reduced_points.dtype}")
    
    # Test greedy reconstruction
    reconstructed_greedy = lift_quadratic(V, W, shift_value, reduced_points)
    rel_error_greedy = np.linalg.norm(reconstructed_greedy - X_pulse) / np.linalg.norm(X_pulse)
    print(f"Greedy QM relative error: {rel_error_greedy:.2e}")

#%%
    # Replace the PyTorch training section with:
    model, trained_state, losses = run_jax_training_example(
        V, W, shift_value, reduced_points, X_pulse, gamma
    )
    
    
# %%
