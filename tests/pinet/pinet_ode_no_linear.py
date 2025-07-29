import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os
from scipy.integrate import solve_ivp

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sparsemodesnet.pinet import PiNetCCP, PiNetNCP, PiNetNCPSkip, ProdPoly

def create_stable_polynomial_without_linear_ode(input_dim: int, max_order: int, seed: int = 42) -> List[np.ndarray]:
    """
    Create stable polynomial ODE coefficients of the form:
    xdot = f(x) = A2*kron(x,x) + A3*kron(kron(x,x),x) + ...
    
    Creates richer dynamics by using coupling between states and balanced scaling.
    
    Parameters
    ----------
    input_dim : int
        State dimension
    max_order : int
        Maximum polynomial order
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    coeffs : List[np.ndarray]
        List of coefficient matrices [A2, A3, A4, A5...]
        - A2: quadratic coefficients (input_dim, input_dim^2)
        - A3: cubic coefficients (input_dim, input_dim^3)
        etc.
    """
    np.random.seed(seed)
    coeffs = []
    
    assert max_order >= 2, "Maximum order must be at least 2 for polynomial ODE."
    
    for order in range(2, max_order + 1):
        feature_dim = input_dim ** order
        
        # Create structured coefficients for richer dynamics
        A_order = np.zeros((input_dim, feature_dim))
        
        if order == 2:
            # Quadratic terms: create oscillatory and coupling behavior
            # Diagonal terms for self-regulation
            for i in range(input_dim):
                quad_idx = i * input_dim + i  # x_i^2 term
                A_order[i, quad_idx] = -np.random.uniform(0.1, 0.3)  # Self-damping
            
            # Cross terms for coupling (create limit cycles)
            for i in range(input_dim):
                for j in range(input_dim):
                    if i != j:
                        quad_idx = i * input_dim + j  # x_i * x_j term
                        # Create some oscillatory coupling
                        if (i + j) % 2 == 0:
                            A_order[i, quad_idx] = np.random.uniform(-0.2, 0.2)
                        else:
                            A_order[(i + 1) % input_dim, quad_idx] = np.random.uniform(-0.15, 0.15)
        
        elif order == 3:
            # Cubic terms: selective activation for richer behavior
            # Only activate a subset of cubic terms to avoid explosion
            n_active_terms = min(input_dim * 5, feature_dim // 3)
            active_indices = np.random.choice(feature_dim, n_active_terms, replace=False)
            
            for i in range(input_dim):
                for idx in active_indices[:input_dim]:
                    A_order[i, idx] = np.random.uniform(-0.05, 0.05)
        
        else:
            # Higher order terms: very sparse and small
            sparsity = min(0.1, 20.0 / feature_dim)  # Adaptive sparsity
            mask = np.random.random((input_dim, feature_dim)) < sparsity
            A_order = np.random.randn(input_dim, feature_dim) * mask * (0.01 / (order - 1))
        
        coeffs.append(A_order)
    
    return coeffs


def generate_ode_trajectory_data(coeffs: List[np.ndarray], 
                                 input_dim: int,
                                 n_trajectories: int = 50,
                                 t_span: Tuple[float, float] = (0, 10),  # Longer time span
                                 n_points: int = 200,  # More points
                                 noise_level: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate trajectory data from polynomial ODE with better initial conditions.
    
    Parameters
    ----------
    coeffs : List[np.ndarray]
        Polynomial ODE coefficients
    input_dim : int
        State dimension
    n_trajectories : int
        Number of trajectories to simulate
    t_span : Tuple[float, float]
        Time span for integration
    n_points : int
        Number of time points per trajectory
    noise_level : float
        Gaussian noise standard deviation for observations
        
    Returns
    -------
    X : torch.Tensor, shape (n_trajectories * n_points, input_dim)
        State data
    Xdot : torch.Tensor, shape (n_trajectories * n_points, input_dim)
        Time derivative data
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    all_states = []
    all_derivatives = []
    
    for i in range(n_trajectories):
        # Better initial conditions for richer dynamics
        if i < n_trajectories // 3:
            # Some trajectories start from larger initial conditions
            x0 = np.random.randn(input_dim) * 1.0
        elif i < 2 * n_trajectories // 3:
            # Some from medium initial conditions
            x0 = np.random.randn(input_dim) * 0.5
        else:
            # Some from small initial conditions
            x0 = np.random.randn(input_dim) * 0.2
        
        # Add some structure to initial conditions
        if input_dim >= 2:
            # Create some correlated initial conditions
            x0[1] = -x0[0] + np.random.randn() * 0.1
        if input_dim >= 4:
            x0[2] = x0[0] * 0.5 + np.random.randn() * 0.1
            x0[3] = -x0[1] * 0.5 + np.random.randn() * 0.1
        
        # Integrate ODE
        try:
            sol = solve_ivp(
                fun=lambda t, x: polynomial_ode_rhs(t, x, coeffs),
                t_span=t_span,
                y0=x0,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-6,  # Slightly relaxed for stability
                atol=1e-8,
                max_step=0.05  # Smaller max step for better accuracy
            )
            
            if not sol.success:
                print(f"Warning: Integration failed for trajectory {i}")
                continue
                
            # Extract states and compute derivatives
            states = sol.y.T  # (n_points, input_dim)
            derivatives = np.array([
                polynomial_ode_rhs(t, states[j], coeffs) 
                for j, t in enumerate(t_eval)
            ])
            
            # Check for numerical issues
            if np.any(np.isnan(states)) or np.any(np.isinf(states)):
                print(f"Warning: NaN/Inf detected in trajectory {i}")
                continue
                
            # More generous bound for interesting dynamics
            if np.any(np.abs(states) > 50):  # Allow larger excursions
                print(f"Warning: States getting too large in trajectory {i}, skipping")
                continue
            
            # Check for dynamics richness - skip if too static
            state_variation = np.std(states, axis=0)
            if np.all(state_variation < 0.01):
                print(f"Warning: Trajectory {i} too static, skipping")
                continue
            
            all_states.append(states)
            all_derivatives.append(derivatives)
            
        except Exception as e:
            print(f"Error integrating trajectory {i}: {e}")
            continue
    
    if len(all_states) == 0:
        raise ValueError("No successful trajectories generated!")
    
    # Concatenate all trajectory data
    X_np = np.vstack(all_states)  # (total_points, input_dim)
    Xdot_np = np.vstack(all_derivatives)  # (total_points, input_dim)
    
    # Add noise
    if noise_level > 0:
        X_np += np.random.randn(*X_np.shape) * noise_level
        Xdot_np += np.random.randn(*Xdot_np.shape) * noise_level
    
    print(f"Generated {len(all_states)} successful trajectories")
    print(f"Total data points: {X_np.shape[0]}")
    print(f"State range: [{X_np.min():.3f}, {X_np.max():.3f}]")
    print(f"State variation (std): [{np.std(X_np, axis=0).min():.3f}, {np.std(X_np, axis=0).max():.3f}]")
    print(f"Derivative range: [{Xdot_np.min():.3f}, {Xdot_np.max():.3f}]")
    
    return torch.from_numpy(X_np.astype(np.float32)), torch.from_numpy(Xdot_np.astype(np.float32))


def polynomial_ode_rhs(t: float, x: np.ndarray, coeffs: List[np.ndarray]) -> np.ndarray:
    """
    Right-hand side of polynomial ODE: xdot = f(x)
    
    Parameters
    ----------
    t : float
        Time (unused, autonomous system)
    x : np.ndarray, shape (input_dim,)
        State vector
    coeffs : List[np.ndarray]
        Polynomial coefficients
        
    Returns
    -------
    xdot : np.ndarray, shape (input_dim,)
        Time derivative
    """
    input_dim = len(x)
    xdot = np.zeros(input_dim)
    
    # Add terms for each order
    for order, A_order in enumerate(coeffs, start=2):
        # Higher-order terms: compute Kronecker product
        x_kron = x.copy()
        for _ in range(order - 1):
            x_kron = np.kron(x_kron, x)
        xdot += A_order @ x_kron
    
    return xdot


def train_pinet_ode(model: nn.Module, 
                    X_train: torch.Tensor, 
                    Xdot_train: torch.Tensor,
                    X_test: torch.Tensor,
                    Xdot_test: torch.Tensor,
                    epochs: int = 500,
                    lr: float = 1e-3) -> Tuple[List[float], List[float]]:
    """
    Train a Pi-Net model to learn ODE dynamics: xdot = f(x).
    """
    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    model = model.to(device)
    X_train, Xdot_train = X_train.to(device), Xdot_train.to(device)
    X_test, Xdot_test = X_test.to(device), Xdot_test.to(device)
    
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
        Xdot_pred = model(X_train)
        loss = criterion(Xdot_pred, Xdot_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            Xdot_test_pred = model(X_test)
            test_loss = criterion(Xdot_test_pred, Xdot_test)
            test_losses.append(test_loss.item())
        model.train()
        
        lr_schedule.step()  
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:>4}, Train Loss: {loss.item():.4e}, Test Loss: {test_loss.item():.4e}")
    
    return model, train_losses, test_losses

def test_polynomial_ode_order(order: int, input_dim: int = 3, seed: int = 42):
    """
    Test Pi-Net implementations on polynomial ODE of given order.
    """
    print(f"\n{'='*60}")
    print(f"Testing Polynomial ODE Order: {order}")
    print(f"Input dim: {input_dim}")
    print(f"{'='*60}")
    
    # Generate stable polynomial ODE coefficients
    coeffs = create_stable_polynomial_without_linear_ode(input_dim, order, seed)
    
    # Generate trajectory data
    try:
        X_all, Xdot_all = generate_ode_trajectory_data(
            coeffs, input_dim, 
            n_trajectories=50, 
            t_span=(0, 5), 
            n_points=500,
            noise_level=0.0005
        )
        
        # Split into train/test
        n_total = X_all.shape[0]
        n_train = int(0.9 * n_total)
        
        indices = torch.randperm(n_total)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train, Xdot_train = X_all[train_idx], Xdot_all[train_idx]
        X_test, Xdot_test = X_all[test_idx], Xdot_all[test_idx]
        
    except Exception as e:
        print(f"Error generating trajectory data: {e}")
        return {'error': f"Data generation failed: {e}"}
    
    print(f"Train data: {X_train.shape}, Test data: {X_test.shape}")
    
    # Model configurations
    inter_dim = max(input_dim*10, 50)  # More capacity for ODE learning
    
    models_to_test = []
    
    # Test CCP (always available) - now with constant term
    models_to_test.append(('PiNetCCP', PiNetCCP(
        in_dim=input_dim,
        out_dim=input_dim,  # Output dimension = input dimension for ODE
        inter_dim=inter_dim,
        poly_order=order,
        drop_constant=True
    )))
    
    # Test NCP and NCP-Skip - now with constant term
    models_to_test.append(('PiNetNCP', PiNetNCP(
        in_dim=input_dim,
        out_dim=input_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_linear=True,
        drop_constant=True 
    )))
    
    models_to_test.append(('PiNetNCPSkip', PiNetNCPSkip(
        in_dim=input_dim,
        out_dim=input_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_linear=True,
        drop_constant=True
    )))
    
    # Test ProdPoly for quartic case - now with constant term
    if order % 2 == 0 and order != 2:
        models_to_test.append(('ProdPoly-CCP', ProdPoly(
            pinet_class=PiNetCCP,
            in_dim=input_dim,
            out_dim=input_dim,
            inter_dim=inter_dim,
            poly_order=2,
            num_polys=order // 2,
            drop_linear=True,
            drop_constant=True,  # Keep constant term for ODE
            normalize=None,
        )))
        
        models_to_test.append(('ProdPoly-NCP', ProdPoly(
            pinet_class=PiNetNCP,
            in_dim=input_dim,
            out_dim=input_dim,
            inter_dim=inter_dim,
            poly_order=2,
            num_polys=order // 2,
            drop_linear=True,
            drop_constant=True,  # Keep constant term for ODE
            normalize=None,
        )))
        
        models_to_test.append(('ProdPoly-NCPSkip', ProdPoly(
            pinet_class=PiNetNCPSkip,
            in_dim=input_dim,
            out_dim=input_dim,
            inter_dim=inter_dim,
            poly_order=2,
            num_polys=order // 2,
            drop_linear=True,
            drop_constant=True,  # Keep constant term for ODE
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
            model, train_losses, test_losses = train_pinet_ode(
                model, X_train, Xdot_train, X_test, Xdot_test, 
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
                'coeffs': coeffs  # Store for reconstruction
            }
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def plot_ode_results_with_reconstruction(all_results: dict, order_list: List[int]):
    """
    Plot training curves and time trajectories for all models.
    Designed for high-dimensional polynomial ODEs without phase plots.
    """
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # ===== FIGURE 1: Training Curves =====
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    axes1 = axes1.flatten()
    
    for i, order in enumerate(order_list):  # Updated orders
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
        
        ax.set_title(f"Order {order} Polynomial ODE Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/pinet-ode-no-linear/pinet_ode_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===== FIGURE 2: Time Trajectories and Errors =====
    for order in order_list:  # Updated orders
        if order not in all_results:
            continue
            
        results = all_results[order]
        valid_models = [name for name, result in results.items() if 'error' not in result]
        
        if not valid_models:
            continue
            
        n_models = len(valid_models)
        
        # Get input dimension from the first valid model's coefficients
        first_result = next(result for result in results.values() if 'error' not in result)
        coeffs = first_result.get('coeffs')
        if coeffs is None or len(coeffs) == 0:
            continue
        
        # Determine input dimension from coefficient shape
        input_dim = coeffs[0].shape[0]  # Output dimension of first coefficient matrix
        
        # Create subplot: n_models rows, 2 columns (trajectories + errors)
        fig2, axes2 = plt.subplots(n_models, 2, figsize=(20, 5*n_models))
        if n_models == 1:
            axes2 = axes2.reshape(1, -1)
        
        # Generate test trajectory parameters
        x0_test = np.random.randn(input_dim) * 0.1  # Random initial condition scaled for stability
        t_test = np.linspace(0, 5, 150)
        
        for model_idx, model_name in enumerate(valid_models):
            result = results[model_name]
            coeffs = result.get('coeffs')
            model = result.get('model')
            
            if coeffs is None:
                continue
            
            try:
                # Move model to correct device
                model = model.to(device)
                
                # True trajectory
                sol_true = solve_ivp(
                    fun=lambda t, x: polynomial_ode_rhs(t, x, coeffs),
                    t_span=(0, 5),
                    y0=x0_test,
                    t_eval=t_test,
                    method='RK45'
                )
                
                # Learned trajectory
                def learned_rhs(t, x):
                    model.eval()
                    with torch.no_grad():
                        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
                        xdot_tensor = model(x_tensor)
                        return xdot_tensor.cpu().numpy().squeeze()
                
                sol_learned = solve_ivp(
                    fun=learned_rhs,
                    t_span=(0, 5),
                    y0=x0_test,
                    t_eval=t_test,
                    method='RK45'
                )
                
                # Plot trajectories
                ax_traj = axes2[model_idx, 0]
                
                if sol_true.success and sol_learned.success:
                    # Plot all state trajectories
                    for state_idx in range(input_dim):
                        ax_traj.plot(t_test, sol_true.y[state_idx], 
                                   color=f'C{state_idx}', linestyle='-', 
                                   alpha=0.7, linewidth=1.5, 
                                   label=f'True x_{state_idx}' if state_idx < 5 else None)
                        ax_traj.plot(t_test, sol_learned.y[state_idx], 
                                   color=f'C{state_idx}', linestyle='--', 
                                   alpha=0.7, linewidth=1.5,
                                   label=f'Learned x_{state_idx}' if state_idx < 5 else None)
                    
                    # Calculate overall trajectory error
                    if len(sol_true.y[0]) == len(sol_learned.y[0]):
                        total_error = np.mean(np.sqrt(np.sum((sol_true.y - sol_learned.y)**2, axis=0)))
                        max_error = np.max(np.sqrt(np.sum((sol_true.y - sol_learned.y)**2, axis=0)))
                        ax_traj.set_title(f'{model_name}\nMean Error: {total_error:.4e}, Max Error: {max_error:.4e}')
                    else:
                        ax_traj.set_title(f'{model_name}')
                    
                    # Only show legend for first few states to avoid clutter
                    if input_dim <= 10:
                        ax_traj.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax_traj.text(0.02, 0.98, f'Solid: True\nDashed: Learned\n{input_dim} states', 
                                   transform=ax_traj.transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Plot errors
                    ax_error = axes2[model_idx, 1]
                    
                    # State-wise errors
                    for state_idx in range(input_dim):
                        state_error = np.abs(sol_true.y[state_idx] - sol_learned.y[state_idx])
                        ax_error.plot(t_test, state_error, 
                                    color=f'C{state_idx}', alpha=0.7, linewidth=1.5,
                                    label=f'Error x_{state_idx}' if state_idx < 5 else None)
                    
                    # Overall trajectory norm error
                    trajectory_error = np.sqrt(np.sum((sol_true.y - sol_learned.y)**2, axis=0))
                    ax_error.plot(t_test, trajectory_error, 'k-', linewidth=2, 
                                label='||error||', alpha=0.8)
                    
                    ax_error.set_title(f'{model_name} - Errors')
                    ax_error.set_yscale('log')
                    
                    if input_dim <= 10:
                        ax_error.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax_error.text(0.02, 0.98, f'Colored: State errors\nBlack: ||error||\n{input_dim} states', 
                                    transform=ax_error.transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                else:
                    ax_traj.set_title(f'{model_name}\nIntegration Failed')
                    ax_error.set_title(f'{model_name} - Errors\nIntegration Failed')
                
                ax_traj.set_xlabel('Time')
                ax_traj.set_ylabel('State Values')
                ax_traj.grid(True, alpha=0.3)
                
                ax_error.set_xlabel('Time')
                ax_error.set_ylabel('Error')
                ax_error.grid(True, alpha=0.3)
                    
            except Exception as e:
                print(f"Error creating trajectory plot for {model_name}, order {order}: {e}")
                # Hide the subplots for this model if there's an error
                for col in range(2):
                    if model_idx < axes2.shape[0] and col < axes2.shape[1]:
                        axes2[model_idx, col].set_visible(False)
        
        fig2.suptitle(f'Order {order} Polynomial ODE - Trajectories and Errors (Dim={input_dim})', 
                     fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'figures/pinet-ode-no-linear/pinet_ode_trajectories_order_{order}_dim_{input_dim}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    return fig1, fig2

def print_ode_summary_table(all_results: dict, order_list: List[int]):
    """
    Print a summary table of final test errors for ODE learning.
    """
    print(f"\n{'='*80}")
    print("SUMMARY: Final Test MSE Loss (ODE Learning)")
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


# Update the main function to include the new summary plot
def main():
    """
    Main test function - runs comprehensive Pi-Net ODE tests.
    """
    print("Pi-Net ODE Implementation Test Suite")
    print("Testing polynomial ODE learning capabilities")
    
    # Test parameters
    input_dim = 10
    
    all_results = {}
    np.random.seed(1234)
    order_list = [2] 
    
    # Test each polynomial order
    for order in order_list:
        try:
            seed = np.random.randint(0, 10000)  # Random seed for each order
            results = test_polynomial_ode_order(order, input_dim, seed)
            all_results[order] = results
        except Exception as e:
            print(f"Error testing order {order}: {e}")
            all_results[order] = {'error': str(e)}
    
    # Print summary
    print_ode_summary_table(all_results, order_list)
    
    # Plot results with reconstruction
    try:
        plot_ode_results_with_reconstruction(all_results, order_list)
    except Exception as e:
        print(f"Error plotting results: {e}")
    
    print(f"\n{'='*60}")
    print("ODE Test suite completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
