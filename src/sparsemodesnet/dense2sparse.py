import numpy as np
import torch
from torch.utils.data import DataLoader
from .pod import compute_pod_basis
from .model import SparseModesNet
from .dataset import PODReconDataset
from .train import train_sparsemodesnet 
from .config import SparseModesNetConfig

def run_sparsemodesnet_d2s(X_np: np.ndarray, config: SparseModesNetConfig):
    """Dense-to-sparse regularization path with warm-start λ→(1+ε)λ routine."""
    print("\n=== Dense-To-Sparse (default) λ-Path ===")
    
    U_s_np, _, _ = compute_pod_basis(X_np, s=config.s)
    Z_np = U_s_np.T.dot(X_np)

    U_s_tensor = torch.from_numpy(
        U_s_np.astype(np.float32)).to(config.training.device)
    Z_tensor = torch.from_numpy(
        Z_np.T.astype(np.float32)).to(config.training.device)

    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(
        dataset_full, batch_size=config.training.batch_size, 
        shuffle=True, drop_last=False
    )

    lam = config.sparsity.lam0
    prev_nonzero = config.s
    path_history = []
    iter_count = 0

    while True:
        iter_count += 1
        print(f"\n-- Path iteration {iter_count}, λ = {lam:.3e} "
              f"(r(λ) prev = {prev_nonzero})")
        
        model = SparseModesNet(
            pod_basis       = U_s_tensor,
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
        
        history = train_sparsemodesnet(
            model, 
            dataloader_full, 
            config.training.num_epochs, 
            config.training.lr, 
            config.training.optimizer, 
            config.training.device
        )
        omega_opt = model.omega.detach().cpu().numpy()
        nonzero_idxs = np.where(np.abs(omega_opt) > config.sparsity.nonzero_thresh)[0]
        curr_nonzero = len(nonzero_idxs)

        model.eval()
        with torch.no_grad():
            _, x_hat_tensor = model(Z_tensor)
            X_hat_np = x_hat_tensor.cpu().numpy().T
        frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')
        rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')

        path_history.append({
            'lambda': lam,
            'nonzero_count': curr_nonzero,
            'selected_idxs': nonzero_idxs.copy(),
            'error': rel_frob_error,
            'l1_b': np.mean(history['l1_b'])
        })

        print(f"  → at λ={lam:.3e}: r(λ)={curr_nonzero}, "
              f"rel_err={rel_frob_error:.6e}")
        print(f"  → selected modes: {nonzero_idxs.tolist()}")

        if curr_nonzero == 0:
            print("All skip-weights have zeroed out. Stopping path.\n")
            break

        lam = lam * (1.0 + config.sparsity.epsilon)
        prev_nonzero = curr_nonzero

        if iter_count >= config.sparsity.max_iters:
            print(f"Reached max_iters={config.sparsity.max_iters} on λ-path; stopping early.\n")
            break

    return path_history

# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from .pod import compute_pod_basis
# from .model import SparseModesNet
# from .dataset import PODReconDataset
# from .train import train_sparsemodesnet 

# def run_sparsemodesnet_d2s(X_np: np.ndarray,
#                            s: int,
#                            hidden_units: list,
#                            M: float,
#                            nonzero_thresh: float,
#                            lam0: float,
#                            epsilon: float,
#                            network_type: str,
#                            poly_order: int,
#                            num_polys: int,
#                            drop_linear: bool,
#                            drop_constant: bool,
#                            lr: float,
#                            B: int,
#                            max_iters: int,
#                            batch_size: int,
#                            optimizer: str,
#                            device: str):
#     """Dense-to-sparse regularization path with warm-start λ→(1+ε)λ routine."""
#     print("\n=== Dense-To-Sparse (default) λ-Path ===")
    
#     U_s_np, _, _ = compute_pod_basis(X_np, s=s)
#     Z_np = U_s_np.T.dot(X_np)

#     U_s_tensor = torch.from_numpy(U_s_np.astype(np.float32)).to(device)
#     Z_tensor = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)

#     dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
#     dataloader_full = DataLoader(
#         dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)

#     lam = lam0
#     prev_nonzero = s
#     path_history = []
#     iter_count = 0

#     while True:
#         iter_count += 1
#         print(f"\n-- Path iteration {iter_count}, λ = {lam:.3e} "
#               f"(r(λ) prev = {prev_nonzero})")
        
#         model = SparseModesNet(
#             pod_basis=U_s_tensor,
#             input_dim=s,
#             hidden_units=hidden_units,
#             M=M,
#             lam=lam,
#             network_type=network_type,
#             poly_order=poly_order,
#             num_polys=num_polys,
#             drop_linear=drop_linear,
#             drop_constant=drop_constant
#         ).to(device)
        
#         history = train_sparsemodesnet(
#             model, dataloader_full, B, lr, optimizer, device)
#         omega_opt = model.omega.detach().cpu().numpy()
#         nonzero_idxs = np.where(np.abs(omega_opt) > nonzero_thresh)[0]
#         curr_nonzero = len(nonzero_idxs)

#         model.eval()
#         with torch.no_grad():
#             _, x_hat_tensor = model(Z_tensor)
#             X_hat_np = x_hat_tensor.cpu().numpy().T
#         frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')
#         rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')

#         path_history.append({
#             'lambda': lam,
#             'nonzero_count': curr_nonzero,
#             'selected_idxs': nonzero_idxs.copy(),
#             'error': rel_frob_error,
#             'l1_b': np.mean(history['l1_b'])
#         })

#         print(f"  → at λ={lam:.3e}: r(λ)={curr_nonzero}, "
#               f"rel_err={rel_frob_error:.6e}")
#         print(f"  → selected modes: {nonzero_idxs.tolist()}")

#         if curr_nonzero == 0:
#             print("All skip-weights have zeroed out. Stopping path.\n")
#             break

#         lam = lam * (1.0 + epsilon)
#         prev_nonzero = curr_nonzero

#         if iter_count >= max_iters:
#             print(f"Reached max_iters={max_iters} on λ-path; stopping early.\n")
#             break

#     return path_history
