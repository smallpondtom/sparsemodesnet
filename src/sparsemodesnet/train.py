import torch.nn as nn
import torch.optim as optim
import torch.utils.data as DataLoader
from sparsemodesnet.model import SparseModesNet

def train_sparsemodesnet(model: SparseModesNet,
                         dataloader: DataLoader,
                         num_epochs: int,
                         lr: float,
                         optimizer: str,
                         device: str):
    """
    Train for exactly num_epochs at whatever model.lam currently is.
    """
    model.to(device)
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError("Unsupported optimizer. Use 'Adam' or 'SGD'.")
    mse_loss = nn.MSELoss()
    lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # learning rate scheduler

    history = {'loss': [], 'l1_b': []}

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_l1  = 0.0
        n_samples = 0

        model.train()
        for z_batch, x_batch in dataloader:
            z_batch = z_batch.to(device)  # (batch, s)
            x_batch = x_batch.to(device)  # (batch, d)

            optimizer.zero_grad()
            z_hat_batch, x_hat_batch = model(z_batch)  # (batch, d)
            loss = mse_loss(x_hat_batch, x_batch)
            loss.backward()
            optimizer.step()

            model.proximal_step()

            batch_size = x_batch.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_l1  += model.l1_norm_b().item() * batch_size
            n_samples += batch_size
            
        lr_schedule.step()  # Update learning rate

        epoch_loss /= n_samples
        epoch_l1  /= n_samples
        history['loss'].append(epoch_loss)
        history['l1_b'].append(epoch_l1)

        # Print every 20 epochs or first:
        if (epoch % 20 == 0) or (epoch == 1):
            print(f"  λ={model.lam:.3e} | Epoch {epoch:3d} | Recon MSE={epoch_loss:.6e} | ‖b‖₁={epoch_l1:.6e}")

    return history