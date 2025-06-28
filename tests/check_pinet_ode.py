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


def create_stable_polynomial_ode(input_dim: int, max_order: int, seed: int = 42) -> List[np.ndarray]:
    """
    Create stable polynomial ODE coefficients of the form:
    xdot = f(x) = a + A*x + A2*kron(x,x) + A3*kron(kron(x,x),x) + ...
    
    Ensures stability by making the linear part (A) stable.
    
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
        List of coefficient matrices [a, A, A2, A3, ...]
        - a: constant term (input_dim, 1)
        - A: linear coefficients (input_dim, input_dim) - stable
        - A2: quadratic coefficients (input_dim, input_dim^2)
        - A3: cubic coefficients (input_dim, input_dim^3)
        etc.
    """
    np.random.seed(seed)
    coeffs = []
    
    # Constant term 
    a = np.random.randn(input_dim, 1) * 0.01  # Small constant term
    coeffs.append(a)
    
    for order in range(1, max_order + 1):
        if order == 1:
            # Create stable linear part: A = -(Q * Q^T) where Q is random
            Q = np.random.randn(input_dim, input_dim) * 0.5
            A = -(Q @ Q.T) - np.eye(input_dim) * 0.1  # Extra damping
            coeffs.append(A)
        else:
            # Higher-order terms (much smaller magnitude)
            feature_dim = input_dim ** order
            A_order = np.random.randn(input_dim, feature_dim) * (0.01 / order)
            coeffs.append(A_order)
    
    return coeffs


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
    for order, A_order in enumerate(coeffs):
        if order == 0:
            # Constant term
            xdot += A_order.flatten() 
        elif order == 1:
            # Linear term: A*x
            xdot += A_order @ x
        else:
            # Higher-order terms: compute Kronecker product
            x_kron = x.copy()
            for _ in range(order - 1):
                x_kron = np.kron(x_kron, x)
            xdot += A_order @ x_kron
    
    return xdot


def generate_ode_trajectory_data(coeffs: List[np.ndarray], 
                                 input_dim: int,
                                 n_trajectories: int = 50,
                                 t_span: Tuple[float, float] = (0, 5),
                                 n_points: int = 100,
                                 noise_level: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate trajectory data from polynomial ODE.
    
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
        # Random initial condition (not too large to avoid instability)
        x0 = np.random.randn(input_dim) * 0.5
        
        # Integrate ODE
        try:
            sol = solve_ivp(
                fun=lambda t, x: polynomial_ode_rhs(t, x, coeffs),
                t_span=t_span,
                y0=x0,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-8,
                atol=1e-10,
                max_step=0.1
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
                
            if np.any(np.abs(states) > 10):  # States getting too large
                print(f"Warning: States getting large in trajectory {i}, skipping")
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
    print(f"Derivative range: [{Xdot_np.min():.3f}, {Xdot_np.max():.3f}]")
    
    return torch.from_numpy(X_np.astype(np.float32)), torch.from_numpy(Xdot_np.astype(np.float32))


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
    coeffs = create_stable_polynomial_ode(input_dim, order, seed)
    
    # Check stability of linear part 
    # (coeffs[1] is the linear part and coeffs[0] is the constant term)
    if len(coeffs) > 1:
        eigenvals = np.linalg.eigvals(coeffs[1])  # Linear part is at index 1
        max_real_part = np.max(np.real(eigenvals))
        print(f"Linear part eigenvalues max real part: {max_real_part:.6f}")
        if max_real_part >= 0:
            print("Warning: Linear part may not be stable!")
    
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
    inter_dim = max(input_dim**4, 100)  # More capacity for ODE learning
    
    models_to_test = []
    
    # Test CCP (always available) - now with constant term
    models_to_test.append(('PiNetCCP', PiNetCCP(
        in_dim=input_dim,
        out_dim=input_dim,  # Output dimension = input dimension for ODE
        inter_dim=inter_dim,
        poly_order=order,
        drop_constant=False  # Keep constant term for ODE
    )))
    
    # Test NCP and NCP-Skip - now with constant term
    models_to_test.append(('PiNetNCP', PiNetNCP(
        in_dim=input_dim,
        out_dim=input_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_linear=False,
        drop_constant=False  # Keep constant term for ODE
    )))
    
    models_to_test.append(('PiNetNCPSkip', PiNetNCPSkip(
        in_dim=input_dim,
        out_dim=input_dim,
        inter_dim=inter_dim,
        poly_order=order,
        drop_linear=False,
        drop_constant=False  # Keep constant term for ODE
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
            drop_linear=False,
            drop_constant=False,  # Keep constant term for ODE
            normalize=None,
        )))
        
        models_to_test.append(('ProdPoly-NCP', ProdPoly(
            pinet_class=PiNetNCP,
            in_dim=input_dim,
            out_dim=input_dim,
            inter_dim=inter_dim,
            poly_order=2,
            num_polys=order // 2,
            drop_linear=False,
            drop_constant=False,  # Keep constant term for ODE
            normalize=None,
        )))
        
        models_to_test.append(('ProdPoly-NCPSkip', ProdPoly(
            pinet_class=PiNetNCPSkip,
            in_dim=input_dim,
            out_dim=input_dim,
            inter_dim=inter_dim,
            poly_order=2,
            num_polys=order // 2,
            drop_linear=False,
            drop_constant=False,  # Keep constant term for ODE
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


# Update all model creation sections in the plotting functions
def plot_ode_results_with_reconstruction(all_results: dict):
    """
    Plot training curves, phase portraits for all models, and time trajectories.
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
        
        ax.set_title(f"Order {order} Polynomial ODE Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/pinet-ode/pinet_ode_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===== FIGURE 2: Phase Portraits for All Models =====
    for order in [1, 2, 3, 4]:
        if order not in all_results:
            continue
            
        results = all_results[order]
        valid_models = [name for name, result in results.items() if 'error' not in result]
        
        if not valid_models:
            continue
            
        n_models = len(valid_models)
        fig2, axes2 = plt.subplots(n_models, 3, figsize=(15, 4*n_models+1))
        if n_models == 1:
            axes2 = axes2.reshape(1, -1)
        
        for model_idx, model_name in enumerate(valid_models):
            result = results[model_name]
            coeffs = result.get('coeffs')
            model = result.get('model')
            
            if coeffs is None:
                continue
            
            try:
                # Move model to correct device
                model = model.to(device)
                
                # Generate test trajectory for visualization
                x0_test = np.array([0.3, -0.2, 0.1])
                t_test = np.linspace(0, 5, 100)
                
                # True trajectory
                sol_true = solve_ivp(
                    fun=lambda t, x: polynomial_ode_rhs(t, x, coeffs),
                    t_span=(0, 5),
                    y0=x0_test,
                    t_eval=t_test,
                    method='RK45'
                )
                
                # Learned trajectory (integrate learned dynamics)
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
                
                # Plot phase portraits (3 projections)
                projections = [(0, 1), (0, 2), (1, 2)]
                projection_labels = ['x₀ vs x₁', 'x₀ vs x₂', 'x₁ vs x₂']
                
                for col, ((i, j), label) in enumerate(zip(projections, projection_labels)):
                    ax = axes2[model_idx, col]
                    
                    if sol_true.success and sol_learned.success:
                        # Plot true trajectory
                        ax.plot(sol_true.y[i], sol_true.y[j], 'b-', label='True', linewidth=2)
                        ax.plot(sol_learned.y[i], sol_learned.y[j], 'r--', label='Learned', linewidth=2)
                        
                        # Mark initial condition
                        ax.plot(x0_test[i], x0_test[j], 'go', markersize=8, label='IC')
                        
                        # Compute trajectory error
                        if len(sol_true.y[0]) == len(sol_learned.y[0]):
                            traj_error = np.mean(np.sqrt(np.sum((sol_true.y - sol_learned.y)**2, axis=0)))
                            ax.set_title(f'{model_name} - {label}\nTraj Error: {traj_error:.3f}')
                        else:
                            ax.set_title(f'{model_name} - {label}')
                    else:
                        ax.set_title(f'{model_name} - {label}\nIntegration Failed')
                    
                    ax.set_xlabel(f'x_{i}')
                    ax.set_ylabel(f'x_{j}')
                    if col == 0:  # Only show legend on first column
                        ax.legend()
                    ax.grid(True, alpha=0.3)
                    
            except Exception as e:
                print(f"Error creating phase portrait for {model_name}, order {order}: {e}")
                # Hide the subplots for this model if there's an error
                for col in range(3):
                    if model_idx < axes2.shape[0] and col < axes2.shape[1]:
                        axes2[model_idx, col].set_visible(False)
        
        fig2.suptitle(f'Order {order} Polynomial ODE - Phase Portraits (All Models)', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'figures/pinet-ode/pinet_ode_phase_portraits_order_{order}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # ===== FIGURE 3: Time Trajectories for All Models =====
    for order in [1, 2, 3, 4]:
        if order not in all_results:
            continue
            
        results = all_results[order]
        valid_models = [name for name, result in results.items() if 'error' not in result]
        
        if not valid_models:
            continue
            
        n_models = len(valid_models)
        fig3, axes3 = plt.subplots(n_models, 3, figsize=(18, 4*n_models))
        if n_models == 1:
            axes3 = axes3.reshape(1, -1)
        
        for model_idx, model_name in enumerate(valid_models):
            result = results[model_name]
            coeffs = result.get('coeffs')
            model = result.get('model')
            
            if coeffs is None:
                continue
            
            try:
                # Move model to correct device
                model = model.to(device)
                
                # Generate test trajectory for visualization
                x0_test = np.array([0.3, -0.2, 0.1])
                t_test = np.linspace(0, 5, 150) 
                
                # True trajectory
                sol_true = solve_ivp(
                    fun=lambda t, x: polynomial_ode_rhs(t, x, coeffs),
                    t_span=(0, 5),
                    y0=x0_test,
                    t_eval=t_test,
                    method='RK45'
                )
                
                # Learned trajectory (integrate learned dynamics)
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
                
                # Plot time trajectories for each state
                state_labels = ['x₀(t)', 'x₁(t)', 'x₂(t)']
                
                for state_idx in range(3):
                    ax = axes3[model_idx, state_idx]
                    
                    if sol_true.success and sol_learned.success:
                        # Plot true and learned trajectories
                        ax.plot(t_test, sol_true.y[state_idx], 'b-', label='True', linewidth=2)
                        ax.plot(t_test, sol_learned.y[state_idx], 'r--', label='Learned', linewidth=2)
                        
                        # Compute state-wise error
                        if len(sol_true.y[state_idx]) == len(sol_learned.y[state_idx]):
                            state_error = np.mean(np.abs(sol_true.y[state_idx] - sol_learned.y[state_idx]))
                            max_error = np.max(np.abs(sol_true.y[state_idx] - sol_learned.y[state_idx]))
                            ax.set_title(f'{model_name} - {state_labels[state_idx]}\nMean Error: {state_error:.4f}, Max Error: {max_error:.4f}')
                        else:
                            ax.set_title(f'{model_name} - {state_labels[state_idx]}')
                    else:
                        ax.set_title(f'{model_name} - {state_labels[state_idx]}\nIntegration Failed')
                    
                    ax.set_xlabel('Time')
                    ax.set_ylabel(f'x_{state_idx}')
                    if state_idx == 0:  # Only show legend on first column
                        ax.legend()
                    ax.grid(True, alpha=0.3)
                    
            except Exception as e:
                print(f"Error creating time trajectory for {model_name}, order {order}: {e}")
                # Hide the subplots for this model if there's an error
                for state_idx in range(3):
                    if model_idx < axes3.shape[0] and state_idx < axes3.shape[1]:
                        axes3[model_idx, state_idx].set_visible(False)
        
        fig3.suptitle(f'Order {order} Polynomial ODE - Time Trajectories (All Models)', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f'figures/pinet-ode/pinet_ode_time_trajectories_order_{order}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return fig1, fig2, fig3


def plot_ode_summary_comparison(all_results: dict):
    """
    Create a summary comparison plot showing best models for each order.
    """
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Find best model for each order
    best_models = {}
    for order in [1, 2, 3, 4]:
        if order not in all_results:
            continue
            
        results = all_results[order]
        best_loss = float('inf')
        best_model_name = None
        best_coeffs = None
        best_model = None
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
            if result['final_test_loss'] < best_loss:
                best_loss = result['final_test_loss']
                best_model_name = model_name
                best_coeffs = result.get('coeffs')
                best_model = result.get('model')
        
        if best_model_name is not None:
            best_models[order] = {
                'model': best_model,
                'name': best_model_name,
                'coeffs': best_coeffs,
                'loss': best_loss,
            }
    
    if not best_models:
        print("No valid models found for summary comparison.")
        return
    
    # Create comparison plots
    n_orders = len(best_models)
    fig, axes = plt.subplots(n_orders, 4, figsize=(20, 4*n_orders))
    if n_orders == 1:
        axes = axes.reshape(1, -1)
    
    for order_idx, (order, model_info) in enumerate(best_models.items()):
        model_name = model_info['name']
        coeffs = model_info['coeffs']
        model = model_info['model']
        
        try:
            # Generate test trajectory
            x0_test = np.array([0.3, -0.2, 0.1])
            t_test = np.linspace(0, 5, 150)
            
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
            
            if sol_true.success and sol_learned.success:
                # Plot time trajectory for x0
                ax = axes[order_idx, 0]
                ax.plot(t_test, sol_true.y[0], 'b-', label='True', linewidth=2)
                ax.plot(t_test, sol_learned.y[0], 'r--', label='Learned', linewidth=2)
                ax.set_title(f'Order {order}: x₀(t)\n{model_name}')
                ax.set_xlabel('Time')
                ax.set_ylabel('x₀')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Plot phase portrait x0 vs x1
                ax = axes[order_idx, 1]
                ax.plot(sol_true.y[0], sol_true.y[1], 'b-', label='True', linewidth=2)
                ax.plot(sol_learned.y[0], sol_learned.y[1], 'r--', label='Learned', linewidth=2)
                ax.plot(x0_test[0], x0_test[1], 'go', markersize=8, label='IC')
                ax.set_title(f'Phase: x₀ vs x₁')
                ax.set_xlabel('x₀')
                ax.set_ylabel('x₁')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Plot 3D trajectory
                ax = axes[order_idx, 2]
                ax = plt.subplot(n_orders, 4, order_idx*4 + 3, projection='3d')
                ax.plot(sol_true.y[0], sol_true.y[1], sol_true.y[2], 'b-', label='True', linewidth=2)
                ax.plot(sol_learned.y[0], sol_learned.y[1], sol_learned.y[2], 'r--', label='Learned', linewidth=2)
                ax.scatter([x0_test[0]], [x0_test[1]], [x0_test[2]], color='green', s=100, label='IC')
                ax.set_title(f'3D Trajectory')
                ax.set_xlabel('x₀')
                ax.set_ylabel('x₁')
                ax.set_zlabel('x₂')
                ax.legend()
                
                # Plot error over time
                ax = axes[order_idx, 3]
                error_norm = np.sqrt(np.sum((sol_true.y - sol_learned.y)**2, axis=0))
                ax.plot(t_test, error_norm, 'g-', linewidth=2)
                ax.set_title(f'Trajectory Error\nFinal: {error_norm[-1]:.3e}')
                ax.set_xlabel('Time')
                ax.set_ylabel('||x_true - x_learned||')
                ax.grid(True, alpha=0.3)
                
        except Exception as e:
            print(f"Error creating summary for order {order}: {e}")
            # Hide subplots for this order
            for col in range(4):
                if order_idx < axes.shape[0] and col < axes.shape[1]:
                    axes[order_idx, col].set_visible(False)
    
    plt.suptitle('Best Models Summary - Polynomial ODE Learning', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/pinet-ode/pinet_ode_best_models_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def print_ode_summary_table(all_results: dict):
    """
    Print a summary table of final test errors for ODE learning.
    """
    print(f"\n{'='*80}")
    print("SUMMARY: Final Test MSE Loss (ODE Learning)")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Order 1':<12} {'Order 2':<12} {'Order 3':<12} {'Order 4':<12}")
    print(f"{'-'*80}")
    
    # Collect all model names
    all_model_names = set()
    for order_results in all_results.values():
        all_model_names.update(order_results.keys())
    
    for model_name in sorted(all_model_names):
        row = f"{model_name:<20} "
        for order in [1, 2, 3, 4]:
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
    input_dim = 3
    
    all_results = {}
    
    np.random.seed(1234)
    
    # Test each polynomial order
    for order in [1, 2, 3, 4]:
        try:
            seed = np.random.randint(0, 10000)  # Random seed for each order
            results = test_polynomial_ode_order(order, input_dim, seed)
            all_results[order] = results
        except Exception as e:
            print(f"Error testing order {order}: {e}")
            all_results[order] = {'error': str(e)}
    
    # Print summary
    print_ode_summary_table(all_results)
    
    # Plot results with reconstruction
    try:
        plot_ode_results_with_reconstruction(all_results)
        plot_ode_summary_comparison(all_results)
    except Exception as e:
        print(f"Error plotting results: {e}")
    
    print(f"\n{'='*60}")
    print("ODE Test suite completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()