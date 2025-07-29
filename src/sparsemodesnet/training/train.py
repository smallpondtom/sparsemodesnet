import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as DataLoader
from sparsemodesnet.models.model import SparseModesNet, StateDecoder


def train_sparsemodesnet(model: SparseModesNet,
                         dataloader: DataLoader,
                         num_epochs: int,
                         lr: float,
                         optimizer: str,
                         momentum: float,
                         max_num_modes: int,
                         extra_modes: int,
                         device: str):
    """
    Train SparseModesNet for exactly num_epochs at whatever model.lam currently 
    is. 
    """
    # Send to device 
    model.to(device)

    # Select optimizer: Adam or SGD
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
                              nesterov=True)
    else:
        raise ValueError("Unsupported optimizer. Use 'Adam' or 'SGD'.")
    
    # Using MSE loss for loss function
    mse_loss = nn.MSELoss()

    # Learning rate scheduler 
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100,
    )
    lr_new = optimizer.param_groups[0]['lr']

    # Initialize history for loss and l1 norm
    history = {'loss': [], 'l1_b': [], 'omegas': []}

    # Flag to exit early if modes reach desired sparsity
    exit_flag = False

    # Training loop 
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_l1  = 0.0
        n_samples = 0

        model.train()
        for z_batch, x_batch in dataloader:
            z_batch = z_batch.to(device)  # (batch, s)
            x_batch = x_batch.to(device)  # (batch, d)

            optimizer.zero_grad()
            _, x_hat_batch = model(z_batch)  # (batch, d)
            loss = mse_loss(x_hat_batch, x_batch)
            
            loss.backward()
            optimizer.step()

            model.proximal_step(model.lam * lr_new)

            batch_size  = x_batch.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_l1   += model.l1_norm_omega().item() * batch_size
            n_samples  += batch_size
            
        lr_schedule.step(loss)  # Update learning rate
        lr_new = optimizer.param_groups[0]['lr']

        epoch_loss /= n_samples
        epoch_l1  /= n_samples
        history['loss'].append(epoch_loss)
        history['l1_b'].append(epoch_l1)
        
        with torch.no_grad():
            omega_ = model.omega.detach().cpu().numpy()
            history['omegas'].append(omega_)
            nonzero_omegas = np.flatnonzero(omega_)
            nonzero_count = len(nonzero_omegas)

        # Print every 10 epochs or first:
        if (epoch % 10 == 0) or (epoch == 1):
            print(f"  λ={model.lam:.3e} | Epoch {epoch:3d} | lr={lr_new:.4e} | "
                  f"Recon MSE={epoch_loss:.6e} | ‖ω‖₁={epoch_l1:.6e} | "
                  f"Non-zero modes: {nonzero_count}")
            
        if nonzero_count <= max_num_modes + extra_modes:
            print(f"Reached maximum non-zero modes ({max_num_modes}). " 
                  f"Stopping training.")
            exit_flag = True
            break
            
        if epoch_l1 == 0:
            print("All modes have zero weights. Stopping training.")
            exit_flag = True
            break

    return omega_, nonzero_omegas, nonzero_count, history, exit_flag



def train_statedecoder(model: StateDecoder,
                       dataloader: DataLoader,
                       num_epochs: int,
                       lr: float,
                       momentum: float,
                       optimizer: str,
                       device: str):
    """
    Train a StateDecoder model for num_epochs using the specified optimizer.
    """    

    # Send model to device
    model.to(device)

    # Select optimizer: Adam or SGD
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=model.gamma)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, nesterov=True,
            weight_decay=model.gamma)
    else:
        raise ValueError("Unsupported optimizer. Use 'Adam' or 'SGD'.")
    
    # Using MSE loss for loss function
    mse_loss = nn.MSELoss()
    
    # Learning rate scheduler 
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100,
    )

    # Initialize history for loss
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_samples = 0

        model.train()
        for z_batch, x_batch in dataloader:
            z_batch = z_batch.to(device)  # (batch, s)
            x_batch = x_batch.to(device)  # (batch, d)

            optimizer.zero_grad()
            x_hat_batch = model(z_batch)  # (batch, d)
            loss = mse_loss(x_hat_batch, x_batch)
            
            loss.backward()
            optimizer.step()

            batch_size = x_batch.shape[0]
            epoch_loss += loss.item() * batch_size
            n_samples += batch_size
            
        epoch_loss /= n_samples
        loss_history.append(epoch_loss)

        lr_schedule.step(loss)  # Update learning rate
        lr_new = optimizer.param_groups[0]['lr']

        # Print every 10 epochs or first:
        if (epoch % 10 == 0) or (epoch == 1):
            print(f"  Epoch {epoch:3d} | lr={lr_new:.4e} | "
              f"Recon MSE={epoch_loss:.6e}")

    return loss_history