import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sparsemodesnet.models.pinet import PiNetCCP, PiNetNCP, PiNetNCPSkip, ProdPoly


def generate_polynomial_data(
    coeffs: List[np.ndarray], 
    n_samples: int = 1000, 
    input_dim: int = 3,
    noise_level: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate polynomial data of the form:
    f(x) = a + A*x + A2*kron(x,x) + A3*kron(kron(x,x),x) + ...
    
    Arguments
    ---------
    coeffs : List[np.ndarray]
        List of coefficient matrices/vectors [a, A, A2, A3, ...]
        - a: constant vector (output_dim,)
        - A: linear coefficients (output_dim, input_dim)
        - A2: quadratic coefficients (output_dim, input_dim^2)
        - A3: cubic coefficients (output_dim, input_dim^3)
        etc.
    n_samples : int
        Number of samples to generate
    input_dim : int
        Input dimension
    noise_level : float
        Gaussian noise standard deviation
        
    Returns
    -------
    X : torch.Tensor, shape (n_samples, input_dim)
        Input data
    Y : torch.Tensor, shape (n_samples, output_dim)
        Output data
    """
    # Generate random input data
    X = torch.randn(n_samples, input_dim)
    
    # Initialize output
    output_dim = coeffs[0].shape[0]
    Y = torch.zeros(n_samples, output_dim)
    
    # Add constant term
    if len(coeffs) > 0:
        Y += torch.from_numpy(coeffs[0]).float()
    
    # Add linear term
    if len(coeffs) > 1:
        A = torch.from_numpy(coeffs[1]).float()  # (output_dim, input_dim)
        Y += torch.mm(X, A.T)  # (n_samples, output_dim)
    
    # Add higher-order terms
    for order in range(2, len(coeffs)):
        A_order = torch.from_numpy(coeffs[order]).float()
        
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
        Y += torch.mm(X_kron, A_order.T)
    
    # Add noise
    if noise_level > 0:
        Y += torch.randn_like(Y) * noise_level
    
    return X, Y


def create_test_coefficients(input_dim: int, output_dim: int, max_order: int) -> List[np.ndarray]:
    """
    Create random coefficient matrices for polynomial testing.
    """
    np.random.seed(42)  # For reproducibility
    coeffs = []
    
    for order in range(max_order + 1):
        if order == 0:
            # Constant term
            a = np.random.randn(output_dim) * 0.5
            coeffs.append(a)
        else:
            # Higher-order terms
            feature_dim = input_dim ** order
            A = np.random.randn(output_dim, feature_dim) * (0.1 / order)  # Scale down higher orders
            coeffs.append(A)
    
    return coeffs


def train_pinet(model: nn.Module, 
                X_train: torch.Tensor, 
                Y_train: torch.Tensor,
                X_test: torch.Tensor,
                Y_test: torch.Tensor,
                epochs: int = 500,
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    lr_schedule = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=epochs // 5,
        gamma=0.5
    )  # learning rate scheduler
    
    train_losses = []
    test_losses = []
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = criterion(Y_pred, Y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            Y_test_pred = model(X_test)
            test_loss = criterion(Y_test_pred, Y_test)
            test_losses.append(test_loss.item())
        model.train()
        
        lr_schedule.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:>4}, Train Loss: {loss.item():.4e}, Test Loss: {test_loss.item():.4e}")
    
    return model, train_losses, test_losses


def test_polynomial_order(order: int, input_dim: int = 3, output_dim: int = 2):
    """
    Test Pi-Net implementations on polynomial of given order.
    """
    print(f"\n{'='*60}")
    print(f"Testing Polynomial Order: {order}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"{'='*60}")
    
    # Generate polynomial coefficients
    coeffs = create_test_coefficients(input_dim, output_dim, order)
    
    # Generate training and test data
    X_train, Y_train = generate_polynomial_data(coeffs, n_samples=1000, 
                                              input_dim=input_dim, noise_level=0.01)
    X_test, Y_test = generate_polynomial_data(coeffs, n_samples=200, 
                                            input_dim=input_dim, noise_level=0.01)
    
    print(f"Data shapes: X_train {X_train.shape}, Y_train {Y_train.shape}")
    print(f"Data ranges: X_train [{X_train.min():.2f}, {X_train.max():.2f}], "
          f"Y_train [{Y_train.min():.2f}, {Y_train.max():.2f}]")
    
    # Model configurations
    inter_dim = max(input_dim * 2, 8)  # Ensure sufficient capacity
    
    models_to_test = []
    
    # Test CCP (always available)
    models_to_test.append(('PiNetCCP', PiNetCCP(
        in_dim=input_dim,
        out_dim=output_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_constant=False
    )))
    
    # Test NCP and NCP-Skip
    models_to_test.append(('PiNetNCP', PiNetNCP(
        in_dim=input_dim,
        out_dim=output_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_linear=False,
        drop_constant=False
    )))
    
    models_to_test.append(('PiNetNCPSkip', PiNetNCPSkip(
        in_dim=input_dim,
        out_dim=output_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_linear=False,
        drop_constant=False
    )))
    
    # Test ProdPoly for quartic case
    if order % 2 == 0 and order != 2:
        models_to_test.append(('ProdPoly-CCP', ProdPoly(
            pinet_class=PiNetCCP,
            in_dim=input_dim,
            out_dim=output_dim,
            inter_dim=inter_dim,
            poly_order=2,  # Two quadratic blocks
            num_polys=order // 2,
            drop_linear=False,
            drop_constant=False,
            normalize=None,
        )))
        
        models_to_test.append(('ProdPoly-NCP', ProdPoly(
            pinet_class=PiNetNCP,
            in_dim=input_dim,
            out_dim=output_dim,
            inter_dim=inter_dim,
            poly_order=2,  # Two quadratic blocks
            num_polys=order // 2,
            drop_linear=False,
            drop_constant=False,
            normalize=None,
        )))
        
        models_to_test.append(('ProdPoly-NCPSkip', ProdPoly(
            pinet_class=PiNetNCPSkip,
            in_dim=input_dim,
            out_dim=output_dim,
            inter_dim=inter_dim,
            poly_order=2,  # Two quadratic blocks
            num_polys=order // 2,
            drop_linear=False,
            drop_constant=False,
            normalize=None,
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
            
            print(f"Final train loss: {final_train_loss:.6f}")
            print(f"Final test loss: {final_test_loss:.6f}")
            
            results[model_name] = {
                'model': model,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'final_train_loss': final_train_loss,
                'final_test_loss': final_test_loss,
                'total_params': total_params
            }
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def plot_results(all_results: dict):
    """
    Plot training curves for all tested models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    orders = [1, 2, 3, 4]
    
    for i, order in enumerate(orders):
        if order not in all_results:
            continue
            
        ax = axes[i]
        results = all_results[order]
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
                
            epochs = range(len(result['train_losses']))
            ax.semilogy(epochs, result['train_losses'], 
                       label=f"{model_name} (train)", linestyle='-')
            ax.semilogy(epochs, result['test_losses'], 
                       label=f"{model_name} (test)", linestyle='--')
        
        ax.set_title(f"Order {order} Polynomial")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/pinet/pinet_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_table(all_results: dict):
    """
    Print a summary table of final test errors.
    """
    print(f"\n{'='*80}")
    print("SUMMARY: Final Test MSE Loss")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Order 1':<12} {'Order 2':<12} {'Order 3':<12} {'Order 4':<12}")
    print(f"{'-'*80}")
    
    # Collect all model names
    all_model_names = set()
    for order_results in all_results.values():
        all_model_names.update(order_results.keys())
    
    for model_name in sorted(all_model_names):
        row = f"{model_name:<20}"
        for order in [1, 2, 3, 4]:
            if order in all_results and model_name in all_results[order]:
                result = all_results[order][model_name]
                if 'error' in result:
                    row += f"{'ERROR':<12}"
                else:
                    row += f"{result['final_test_loss']:.2e}  "
            else:
                row += f"{'N/A':<12}"
        print(row)
    
    print(f"{'-'*80}")
    

def visualize_polynomial_reconstruction(
    coeffs: List[np.ndarray], 
    model: nn.Module, 
    input_dim: int,
    output_dim: int,
    device: str = 'cpu',
    n_plot_samples: int = 100) -> None:
    """
    Visualize the polynomial reconstruction by comparing true vs predicted outputs.
    
    Parameters
    ----------
    coeffs : List[np.ndarray]
        Polynomial coefficients used to generate true data
    model : nn.Module
        Trained Pi-Net model
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    device : str
        Device for computation
    n_plot_samples : int
        Number of samples for visualization
    """
    # Generate test data for visualization
    X_viz, Y_true = generate_polynomial_data(
        coeffs, n_samples=n_plot_samples, input_dim=input_dim, noise_level=0.0
    )
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        X_viz_device = X_viz.to(device)
        Y_pred = model(X_viz_device).cpu()
    
    # Create subplots for each output dimension
    fig, axes = plt.subplots(1, output_dim, figsize=(5*output_dim, 4))
    if output_dim == 1:
        axes = [axes]
    
    for dim in range(output_dim):
        ax = axes[dim]
        
        # Sort by first input dimension for cleaner visualization
        sort_idx = torch.argsort(X_viz[:, 0])
        x_sorted = X_viz[sort_idx, 0].numpy()
        y_true_sorted = Y_true[sort_idx, dim].numpy()
        y_pred_sorted = Y_pred[sort_idx, dim].numpy()
        
        # Plot true vs predicted
        ax.scatter(x_sorted, y_true_sorted, alpha=0.6, label='True', s=20)
        ax.scatter(x_sorted, y_pred_sorted, alpha=0.6, label='Predicted', s=20)
        
        # Compute and display error metrics
        mse = torch.mean((Y_true[:, dim] - Y_pred[:, dim])**2).item()
        r2 = 1 - torch.sum((Y_true[:, dim] - Y_pred[:, dim])**2) / torch.sum((Y_true[:, dim] - torch.mean(Y_true[:, dim]))**2)
        
        ax.set_xlabel(f'Input Dimension 0')
        ax.set_ylabel(f'Output Dimension {dim}')
        ax.set_title(f'Output {dim}\nMSE: {mse:.4f}, R²: {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_results_with_reconstruction(all_results: dict, coeffs_dict: dict):
    """
    Plot training curves and reconstruction visualizations for all tested models.
    Creates separate figures for training curves and reconstruction plots.
    """
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # ===== FIGURE 1: Training Curves =====
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    axes1 = axes1.flatten()
    
    for i, order in enumerate([1, 2, 3, 4]):
        if order not in all_results:
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
        
        ax.set_title(f"Order {order} Polynomial Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/pinet/pinet_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===== FIGURE 2: Reconstruction Plots =====
    fig2, axes2 = plt.subplots(4, 2, figsize=(12, 16))
    
    for order_idx, order in enumerate([1, 2, 3, 4]):
        if order not in all_results or order not in coeffs_dict:
            # Hide unused subplots
            for dim in range(2):
                axes2[order_idx, dim].set_visible(False)
            continue
            
        results = all_results[order]
        
        # Find the best performing model for this order
        best_model_name = None
        best_loss = float('inf')
        best_model = None
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
            if result['final_test_loss'] < best_loss:
                best_loss = result['final_test_loss']
                best_model_name = model_name
                best_model = result['model']
        
        if best_model_name is None:
            # Hide unused subplots
            for dim in range(2):
                axes2[order_idx, dim].set_visible(False)
            continue
            
        # Create the model again (since we didn't save it in the results)
        input_dim, output_dim = 3, 2
        inter_dim = max(input_dim * 2, 8)
        
        try:
            # Retrain the best model quickly for visualization
            coeffs_order = create_test_coefficients(input_dim, output_dim, order)
            
            # Generate visualization data
            X_viz, Y_true = generate_polynomial_data(coeffs_order, n_samples=100, input_dim=input_dim, noise_level=0.0)
            
            best_model.eval()
            with torch.no_grad():
                X_viz_device = X_viz.to(device)
                Y_pred = best_model(X_viz_device).cpu()
            
            # Plot reconstructions for each output dimension
            for dim in range(output_dim):
                ax = axes2[order_idx, dim]
                
                # Sort by first input dimension for cleaner visualization
                sort_idx = torch.argsort(X_viz[:, 0])
                x_sorted = X_viz[sort_idx, 0].numpy()
                y_true_sorted = Y_true[sort_idx, dim].numpy()
                y_pred_sorted = Y_pred[sort_idx, dim].numpy()
                
                # Plot true vs predicted
                ax.scatter(x_sorted, y_true_sorted, alpha=0.7, label='True', s=25, color='blue')
                ax.scatter(x_sorted, y_pred_sorted, alpha=0.7, label='Predicted', s=25, color='red', marker='x')
                
                # Compute and display error metrics
                mse = torch.mean((Y_true[:, dim] - Y_pred[:, dim])**2).item()
                r2 = 1 - torch.sum((Y_true[:, dim] - Y_pred[:, dim])**2) / torch.sum((Y_true[:, dim] - torch.mean(Y_true[:, dim]))**2)
                
                ax.set_xlabel(f'x₀')
                ax.set_ylabel(f'y_{dim}')
                ax.set_title(f'Order {order}, Output Dim {dim}\n{best_model_name}\nMSE: {mse:.1e}, R²: {r2:.3f}')
                
                if dim == 0:  # Only show legend on first column
                    ax.legend()
                ax.grid(True, alpha=0.3)
                
        except Exception as e:
            print(f"Error creating reconstruction plot for order {order}: {e}")
            # Hide the subplots for this order if there's an error
            for dim in range(2):
                axes2[order_idx, dim].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('figures/pinet/pinet_reconstruction_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig1, fig2

def main():
    """
    Main test function - runs comprehensive Pi-Net tests.
    """
    print("Pi-Net Implementation Test Suite")
    print("Testing polynomial approximation capabilities")
    
    # Test parameters
    input_dim = 3
    output_dim = 2
    
    all_results = {}
    coeffs_dict = {}  # Store coefficients for reconstruction plots
    
    # Test each polynomial order
    for order in [1, 2, 3, 4]:
        try:
            # Store coefficients for later reconstruction visualization
            coeffs_dict[order] = create_test_coefficients(input_dim, output_dim, order)
            
            results = test_polynomial_order(order, input_dim, output_dim)
            all_results[order] = results
        except Exception as e:
            print(f"Error testing order {order}: {e}")
            all_results[order] = {'error': str(e)}
    
    # Print summary
    print_summary_table(all_results)
    
    # Plot results with reconstruction
    try:
        plot_results_with_reconstruction(all_results, coeffs_dict)
    except Exception as e:
        print(f"Error plotting results: {e}")
        # Fallback to original plotting
        try:
            plot_results(all_results)
        except Exception as e2:
            print(f"Error with fallback plotting: {e2}")
    
    print(f"\n{'='*60}")
    print("Test suite completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()