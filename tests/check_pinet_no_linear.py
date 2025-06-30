import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sparsemodesnet.pinet import PiNetCCP, PiNetNCP, PiNetNCPSkip, ProdPoly


def generate_polynomial_data_no_linear(
    coeffs: List[np.ndarray], 
    n_samples: int = 1000, 
    input_dim: int = 10,
    output_dim: int = 5,
    noise_level: float = 0.01,
    input_range: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate polynomial data of the form (no constant or linear terms):
    f(x) = A2*kron(x,x) + A3*kron(kron(x,x),x) + ...
    
    Arguments
    ---------
    coeffs : List[np.ndarray]
        List of coefficient matrices [A2, A3, A4, ...]
        - A2: quadratic coefficients (output_dim, input_dim^2)
        - A3: cubic coefficients (output_dim, input_dim^3)
        etc.
    n_samples : int
        Number of samples to generate
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    noise_level : float
        Gaussian noise standard deviation
    input_range : float
        Range for input data generation
        
    Returns
    -------
    X : torch.Tensor, shape (n_samples, input_dim)
        Input data
    Y : torch.Tensor, shape (n_samples, output_dim)
        Output data
    """
    # Generate random input data with specified range
    X = torch.randn(n_samples, input_dim) * input_range
    
    # Initialize output
    Y = torch.zeros(n_samples, output_dim)
    
    # Add polynomial terms starting from order 2 (quadratic)
    for order_idx, A_order in enumerate(coeffs):
        order = order_idx + 2  # Start from order 2
        A_order_tensor = torch.from_numpy(A_order).float()
        
        # Compute Kronecker product iteratively for each sample
        X_kron_list = []
        for i in range(n_samples):
            x_sample = X[i]  # (input_dim,)
            x_kron = x_sample.clone()
            
            # Build kron(kron(...kron(x,x)...,x),x) with (order-1) applications
            for _ in range(order - 1):
                x_kron = torch.kron(x_kron, x_sample)
            
            X_kron_list.append(x_kron)
        
        X_kron = torch.stack(X_kron_list)  # (n_samples, input_dim^order)
        
        # Apply coefficients
        Y += torch.mm(X_kron, A_order_tensor.T)
    
    # Add noise
    if noise_level > 0:
        Y += torch.randn_like(Y) * noise_level
    
    return X, Y


def create_polynomial_coefficients_no_linear(input_dim: int, output_dim: int, max_order: int, seed: int = 42) -> List[np.ndarray]:
    """
    Create polynomial coefficients without constant and linear terms.
    Similar structure to check_pinet_ode_no_linear.py but for function approximation.
    
    Parameters
    ----------
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    max_order : int
        Maximum polynomial order
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    coeffs : List[np.ndarray]
        List of coefficient matrices [A2, A3, A4, A5...]
        - A2: quadratic coefficients (output_dim, input_dim^2)
        - A3: cubic coefficients (output_dim, input_dim^3)
        etc.
    """
    np.random.seed(seed)
    coeffs = []
    
    assert max_order >= 2, "Maximum order must be at least 2 for polynomial without linear terms."
    
    for order in range(2, max_order + 1):
        feature_dim = input_dim ** order
        
        # Create structured coefficients for richer but stable behavior
        A_order = np.zeros((output_dim, feature_dim))
        
        if order == 2:
            # Quadratic terms: create coupling between input and output dimensions
            # Diagonal coupling: each output influenced by specific input combinations
            for out_idx in range(output_dim):
                for in_i in range(input_dim):
                    for in_j in range(input_dim):
                        quad_idx = in_i * input_dim + in_j  # x_i * x_j term
                        
                        # Create structured coupling patterns
                        if in_i == in_j:  # Diagonal terms (x_i^2)
                            # Each output has stronger coupling to certain input squares
                            if (in_i + out_idx) % input_dim < output_dim:
                                A_order[out_idx, quad_idx] = np.random.uniform(-0.5, 0.5)
                        else:  # Cross terms (x_i * x_j)
                            # Sparser cross-coupling
                            if np.random.random() < 0.3:  # 30% sparsity
                                A_order[out_idx, quad_idx] = np.random.uniform(-0.3, 0.3)
        
        elif order == 3:
            # Cubic terms: selective activation for richer behavior
            sparsity = min(0.1, 50.0 / feature_dim)  # Adaptive sparsity
            for out_idx in range(output_dim):
                n_active_terms = max(1, int(feature_dim * sparsity))
                active_indices = np.random.choice(feature_dim, n_active_terms, replace=False)
                
                for idx in active_indices:
                    A_order[out_idx, idx] = np.random.uniform(-0.1, 0.1)
        
        else:
            # Higher order terms: very sparse and small
            sparsity = min(0.05, 20.0 / feature_dim)  # Very sparse for high orders
            mask = np.random.random((output_dim, feature_dim)) < sparsity
            A_order = np.random.randn(output_dim, feature_dim) * mask * (0.02 / (order - 1))
        
        coeffs.append(A_order)
    
    return coeffs


def train_pinet(model: nn.Module, 
                X_train: torch.Tensor, 
                Y_train: torch.Tensor,
                X_test: torch.Tensor,
                Y_test: torch.Tensor,
                epochs: int = 1000,
                lr: float = 1e-3) -> Tuple[List[float], List[float]]:
    """
    Train a Pi-Net model and return training history.
    """
    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    model = model.to(device)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # More aggressive learning rate schedule for complex polynomials
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=200,
    )
    
    train_losses = []
    test_losses = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = criterion(Y_pred, Y_train)
        loss.backward()
        
        # Gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            Y_test_pred = model(X_test)
            test_loss = criterion(Y_test_pred, Y_test)
            test_losses.append(test_loss.item())
        model.train()
        
        lr_schedule.step(test_loss)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:>4}, Train Loss: {loss.item():.4e}, Test Loss: {test_loss.item():.4e}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return model, train_losses, test_losses


def test_polynomial_order_no_linear(order: int, input_dim: int = 10, output_dim: int = 5, seed: int = 42):
    """
    Test Pi-Net implementations on polynomial of given order without linear terms.
    """
    print(f"\n{'='*60}")
    print(f"Testing Polynomial Order: {order} (No Linear Terms)")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"{'='*60}")
    
    # Generate polynomial coefficients (no constant, no linear)
    coeffs = create_polynomial_coefficients_no_linear(input_dim, output_dim, order, seed)
    
    # Generate training and test data
    try:
        X_train, Y_train = generate_polynomial_data_no_linear(
            coeffs, n_samples=2000, input_dim=input_dim, output_dim=output_dim,
            noise_level=0.005, input_range=0.8
        )
        X_test, Y_test = generate_polynomial_data_no_linear(
            coeffs, n_samples=500, input_dim=input_dim, output_dim=output_dim,
            noise_level=0.005, input_range=0.8
        )
    except Exception as e:
        print(f"Error generating data: {e}")
        return {'error': f"Data generation failed: {e}"}
    
    print(f"Data shapes: X_train {X_train.shape}, Y_train {Y_train.shape}")
    print(f"Data ranges: X_train [{X_train.min():.2f}, {X_train.max():.2f}], "
          f"Y_train [{Y_train.min():.2f}, {Y_train.max():.2f}]")
    
    # Model configurations - larger capacity for high-dimensional polynomial
    inter_dim = max(input_dim * 8, 100)  # Increased capacity
    
    models_to_test = []
    
    # Test CCP (always available) - drop constant and linear
    models_to_test.append(('PiNetCCP', PiNetCCP(
        in_dim=input_dim,
        out_dim=output_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_constant=True,
        normalize='last'
    )))
    
    # Test NCP and NCP-Skip - drop constant and linear
    models_to_test.append(('PiNetNCP', PiNetNCP(
        in_dim=input_dim,
        out_dim=output_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_linear=True,
        drop_constant=True,
        normalize='last'
    )))
    
    models_to_test.append(('PiNetNCPSkip', PiNetNCPSkip(
        in_dim=input_dim,
        out_dim=output_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_linear=True,
        drop_constant=True,
        normalize='last'
    )))
    
    # Test ProdPoly for even orders
    if order % 2 == 0 and order >= 4:
        models_to_test.append(('ProdPoly-CCP', ProdPoly(
            pinet_class=PiNetCCP,
            in_dim=input_dim,
            out_dim=output_dim,
            inter_dim=inter_dim,
            poly_order=2,
            num_polys=order // 2,
            drop_linear=True,
            drop_constant=True,
            normalize='last',
        )))
        
        models_to_test.append(('ProdPoly-NCP', ProdPoly(
            pinet_class=PiNetNCP,
            in_dim=input_dim,
            out_dim=output_dim,
            inter_dim=inter_dim,
            poly_order=2,
            num_polys=order // 2,
            drop_linear=True,
            drop_constant=True,
            normalize='last',
        )))
        
        models_to_test.append(('ProdPoly-NCPSkip', ProdPoly(
            pinet_class=PiNetNCPSkip,
            in_dim=input_dim,
            out_dim=output_dim,
            inter_dim=inter_dim,
            poly_order=2,
            num_polys=order // 2,
            drop_linear=True,
            drop_constant=True,
            normalize='last',
        )))
    
    # Train and evaluate each model
    results = {}
    for model_name, model in models_to_test:
        print(f"\n--- Training {model_name} ---")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params}")
        
        try:
            model, train_losses, test_losses = train_pinet(
                model, X_train, Y_train, X_test, Y_test, 
                epochs=3000, lr=1e-3
            )
            
            final_train_loss = train_losses[-1]
            final_test_loss = test_losses[-1]
            
            print(f"Final train loss: {final_train_loss:.4e}")
            print(f"Final test loss: {final_test_loss:.4e}")
            
            results[model_name] = {
                'model': model,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'final_train_loss': final_train_loss,
                'final_test_loss': final_test_loss,
                'total_params': total_params,
                'coeffs': coeffs  # Store for analysis
            }
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def plot_results_no_linear(all_results: dict, order_list: List[int]):
    """
    Plot training curves and approximation quality for high-dimensional polynomial models.
    """
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # ===== FIGURE 1: Training Curves =====
    n_orders = len(order_list)
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    axes1 = axes1.flatten()
    
    for i, order in enumerate(order_list):
        if i >= 4 or order not in all_results:
            if i < 4:
                axes1[i].set_visible(False)
            continue
            
        ax = axes1[i]
        results = all_results[order]
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
                
            epochs = range(len(result['train_losses']))
            ax.semilogy(epochs, result['train_losses'], 
                       label=f"{model_name} (train)", linestyle='-', alpha=0.8)
            ax.semilogy(epochs, result['test_losses'], 
                       label=f"{model_name} (test)", linestyle='--', alpha=0.8)
        
        ax.set_title(f"Order {order} Polynomial (No Linear Terms)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/pinet-no-linear/pinet_no_linear_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===== FIGURE 2: Approximation Quality Analysis =====
    for order in order_list:
        if order not in all_results:
            continue
            
        results = all_results[order]
        valid_models = [name for name, result in results.items() if 'error' not in result]
        
        if not valid_models:
            continue
            
        n_models = len(valid_models)
        
        # Get dimensions from the first valid model's coefficients
        first_result = next(result for result in results.values() if 'error' not in result)
        coeffs = first_result.get('coeffs')
        if coeffs is None or len(coeffs) == 0:
            continue
        
        input_dim = coeffs[0].shape[1] ** (1/2)  # Approximate from quadratic term
        input_dim = int(round(input_dim))
        output_dim = coeffs[0].shape[0]
        
        # Create test data for approximation analysis
        try:
            X_test_viz, Y_true_viz = generate_polynomial_data_no_linear(
                coeffs, n_samples=300, input_dim=input_dim, output_dim=output_dim,
                noise_level=0.0, input_range=0.8
            )
        except Exception as e:
            print(f"Error generating visualization data for order {order}: {e}")
            continue
        
        # Create subplot: n_models rows, 2 columns (predictions + errors)
        fig2, axes2 = plt.subplots(n_models, 2, figsize=(16, 4*n_models))
        if n_models == 1:
            axes2 = axes2.reshape(1, -1)
        
        for model_idx, model_name in enumerate(valid_models):
            result = results[model_name]
            model = result.get('model')
            
            if model is None:
                continue
            
            try:
                # Move model to correct device and get predictions
                model = model.to(device)
                model.eval()
                with torch.no_grad():
                    X_test_device = X_test_viz.to(device)
                    Y_pred_viz = model(X_test_device).cpu()
                
                # Plot predictions vs true values
                ax_pred = axes2[model_idx, 0]
                
                # For high-dimensional output
                dims_to_plot = output_dim
                for dim in range(dims_to_plot):
                    ax_pred.scatter(Y_true_viz[:, dim].numpy(), Y_pred_viz[:, dim].numpy(), 
                                  alpha=0.6, label=f'Dim {dim}', s=20)
                
                # Perfect prediction line
                y_range = [min(Y_true_viz.min(), Y_pred_viz.min()), 
                          max(Y_true_viz.max(), Y_pred_viz.max())]
                ax_pred.plot(y_range, y_range, 'k--', alpha=0.5, label='Perfect')
                
                # Compute overall R²
                mse = torch.mean((Y_true_viz - Y_pred_viz)**2).item()
                total_var = torch.var(Y_true_viz).item()
                r2_overall = 1 - mse / total_var if total_var > 0 else 0
                
                ax_pred.set_xlabel('True Values')
                ax_pred.set_ylabel('Predicted Values')
                ax_pred.set_title(f'{model_name}\nR²: {r2_overall:.3f}, MSE: {mse:.3e}')
                ax_pred.legend()
                ax_pred.grid(True, alpha=0.3)
                
                # Plot error distribution
                ax_error = axes2[model_idx, 1]
                
                errors = (Y_true_viz - Y_pred_viz).numpy()
                # Show error distribution for each output dimension
                for dim in range(output_dim):
                    ax_error.hist(errors[:, dim], bins=30, alpha=0.6, 
                                label=f'Dim {dim}', density=True)
                
                ax_error.set_xlabel('Prediction Error')
                ax_error.set_ylabel('Density')
                ax_error.set_title(f'{model_name} - Error Distribution')
                ax_error.legend()
                ax_error.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Error creating visualization for {model_name}, order {order}: {e}")
                # Hide the subplots for this model if there's an error
                for col in range(2):
                    if model_idx < axes2.shape[0] and col < axes2.shape[1]:
                        axes2[model_idx, col].set_visible(False)
        
        fig2.suptitle(f'Order {order} Polynomial Approximation Quality (Input Dim={input_dim}, Output Dim={output_dim})', 
                     fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'figures/pinet-no-linear/pinet_no_linear_approx_order_{order}_dim_{input_dim}x{output_dim}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    return fig1, fig2


def print_summary_table_no_linear(all_results: dict, order_list: List[int]):
    """
    Print a summary table of final test errors for polynomial approximation without linear terms.
    """
    print(f"\n{'='*80}")
    print("SUMMARY: Final Test MSE Loss (Polynomial Approximation - No Linear Terms)")
    print(f"{'='*80}")
    
    # Dynamic header based on actual order_list
    header = f"{'Model':<20} "
    for order in order_list:
        header += f"{'Order ' + str(order):<12} "
    print(header)
    print(f"{'-'*80}")
    
    # Collect all model names
    all_model_names = set()
    for order_results in all_results.values():
        all_model_names.update(order_results.keys())
    
    for model_name in sorted(all_model_names):
        row = f"{model_name:<20} "
        for order in order_list:
            if order in all_results and model_name in all_results[order]:
                result = all_results[order][model_name]
                if 'error' in result:
                    row += f"{'ERROR':<12} "
                else:
                    row += f"{result['final_test_loss']:<12.2e} "
            else:
                row += f"{'N/A':<12} "
        print(row)
    
    print(f"{'-'*80}")


def main():
    """
    Main test function - runs comprehensive Pi-Net tests for polynomials without linear terms.
    """
    print("Pi-Net Implementation Test Suite")
    print("Testing polynomial approximation capabilities (No Constant/Linear Terms)")
    
    # Test parameters - higher dimensional
    input_dim = 10
    output_dim = 8
    
    all_results = {}
    np.random.seed(1234)
    order_list = [2, 3, 4, 5]  # Start from order 2 since no linear terms
    
    # Test each polynomial order
    for order in order_list:
        try:
            seed = np.random.randint(0, 10000)  # Random seed for each order
            results = test_polynomial_order_no_linear(order, input_dim, output_dim, seed)
            all_results[order] = results
        except Exception as e:
            print(f"Error testing order {order}: {e}")
            all_results[order] = {'error': str(e)}
    
    # Print summary
    print_summary_table_no_linear(all_results, order_list)
    
    # Plot results
    try:
        plot_results_no_linear(all_results, order_list)
    except Exception as e:
        print(f"Error plotting results: {e}")
    
    print(f"\n{'='*60}")
    print("Test suite completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()