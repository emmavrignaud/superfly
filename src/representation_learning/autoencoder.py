"""
Dense autoencoder utilities for per-fly vector representations.
"""

from __future__ import annotations

import numpy as np


def fit_autoencoder_latent(
    X: np.ndarray,
    latent_dim: int = 64,
    hidden_dims: tuple[int, int] = (512, 256),
    epochs: int = 300,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    val_fraction: float = 0.2,
    patience: int = 30,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Train a dense autoencoder and return latent vectors for all rows.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise ImportError(
            "PyTorch is required for autoencoder latent learning. "
            "Install with `pip install torch` or add it to environment.yml."
        ) from exc

    if X.ndim != 2 or len(X) < 4:
        return X, {"used_autoencoder": False, "reason": "insufficient_data"}

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(X) * val_fraction)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) < 2:
        return X, {"used_autoencoder": False, "reason": "insufficient_train_rows"}

    x_train = torch.tensor(X[train_idx], dtype=torch.float32)
    x_val = torch.tensor(X[val_idx], dtype=torch.float32)
    x_all = torch.tensor(X, dtype=torch.float32)

    in_dim = X.shape[1]
    h1, h2 = hidden_dims
    latent_dim = int(min(max(2, latent_dim), max(2, in_dim - 1)))

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, h1),
                nn.ReLU(),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Linear(h2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, h2),
                nn.ReLU(),
                nn.Linear(h2, h1),
                nn.ReLU(),
                nn.Linear(h1, in_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat, z

    torch.manual_seed(seed)
    model = AE()
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    stale = 0
    history: list[tuple[int, float, float]] = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for (xb,) in train_loader:
            opt.zero_grad()
            xh, _ = model(xb)
            loss = loss_fn(xh, xb)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            xv_hat, _ = model(x_val)
            val_loss = float(loss_fn(xv_hat, x_val).item())
        tr_loss = float(np.mean(tr_losses)) if tr_losses else np.nan
        history.append((ep, tr_loss, val_loss))

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            stale = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        _, z_all = model(x_all)
    Z = z_all.cpu().numpy()
    if verbose:
        print(
            f"[autoencoder] trained epochs={len(history)} latent_dim={latent_dim} "
            f"best_val_mse={best_val:.6g}"
        )

    return Z, {
        "used_autoencoder": True,
        "latent_dim": latent_dim,
        "epochs_trained": len(history),
        "best_val_mse": best_val,
    }
