import numpy as np
import torch
from torch.utils.data import DataLoader
from sparsemodesnet.linalg.pod import compute_pod_basis
from sparsemodesnet.models.model import SparseModesNet
from sparsemodesnet.dataset import PODReconDataset
from sparsemodesnet.config import SparseModesNetConfig
from .train import train_sparsemodesnet 

def dense2sparse(X_np: np.ndarray, Z_np: np.ndarray, U_np: np.ndarray,
                 config: SparseModesNetConfig):
    """Dense-to-sparse regularization path with warm-start λ→(1+ε)λ routine."""
    print("\n=== Dense-To-Sparse (default) λ-Path ===")

    # Convert to PyTorch tensor
    U_tensor = torch.from_numpy(
        U_np.astype(np.float32)).to(config.training.device)

    # Create the dataset and dataloader
    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(
        dataset_full, batch_size=config.training.batch_size, 
        shuffle=True, drop_last=False
    )

    # Initialize the regularization path
    lam = config.sparsity.lam0
    prev_nonzero = config.s
    path_history = []
    iter_count = 0

    # Initialize the SparseModesNet model 
    model = SparseModesNet(
        pod_basis       = U_tensor,
        input_dim       = config.s,
        hidden_units    = config.network.hidden_units,
        M               = config.sparsity.M,
        lam             = lam,
        network_type    = config.network.network_type,
        poly_order      = config.network.poly_order,
        num_polys       = config.network.num_polys,
        drop_linear     = config.network.drop_linear,
        drop_constant   = config.network.drop_constant
    ).to(config.training.device)

    # Define the storage for omegas
    omegas = model.omega.detach().numpy().reshape(-1, 1)

    # Add tracking variables before the while loop
    prev_num_selected = None
    no_change_iterations = 0

    while True:
        iter_count += 1
        print(f"\n-- Path iteration {iter_count}, λ = {lam:.3e} "
              f"(r(λ) prev = {prev_nonzero})")

        # Traing the SparseModesNet model for the current λ 
        omega_, nonzero_idxs, num_selected, history, exit_flag = train_sparsemodesnet(
            model, 
            dataloader_full, 
            config.training.num_epochs, 
            config.training.lr, 
            config.training.optimizer, 
            config.training.momentum,
            config.sparsity.max_num_modes,
            config.training.device
        )

        # Check for convergence based on number of selected modes
        if prev_num_selected is None:
            prev_num_selected = num_selected
            no_change_iterations = 0
        elif num_selected == prev_num_selected:
            no_change_iterations += 1
            print(f"Number of selected modes unchanged for {no_change_iterations}"
                  f" consecutive iterations.")
        else:
            prev_num_selected = num_selected
            no_change_iterations = 0

        # Store current omegas
        omegas = np.concatenate((omegas, omega_.reshape(-1, 1)), axis=1)
        
        if exit_flag:
            break
            
        # Break if number of selected modes hasn't changed for several iterations
        if no_change_iterations >= config.training.max_no_change:
            print(f"Number of selected modes unchanged for {config.training.max_no_change} "
                  f"consecutive iterations. Assuming convergence.")
            break

        path_history.append({
            'lambda': lam,
            'nonzero_count': num_selected,
            'selected_idxs': nonzero_idxs.copy(),
            'l1_b': np.mean(history['l1_b'])
        })

        print(f"  → at λ={lam:.3e}: r(λ)={num_selected}")
        print(f"  → selected modes: {nonzero_idxs.tolist()}")

        lam = lam * (1.0 + config.sparsity.epsilon)

        if iter_count >= config.sparsity.max_iters:
            print(f"Reached max_iters={config.sparsity.max_iters} on λ-path; stopping early.\n")
            break

    # Select the modes with the highest final weights
    I_nn = np.argsort(omegas[:, -1])[::-1][:config.training.max_num_modes]
    print(f"\nFinal selected modes (indices): {I_nn.tolist()}")

    return I_nn, omegas, path_history
