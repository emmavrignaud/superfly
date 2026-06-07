"""
Dense autoencoder utilities for per-fly vector representations.
Implemented in pure numpy + scipy — no PyTorch dependency.

fit_autoencoder_latent    — reconstruction-only
fit_multitask_autoencoder — reconstruction + age classification head
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internals: numpy Adam + MLP layers
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_back(grad: np.ndarray, pre: np.ndarray) -> np.ndarray:
    return grad * (pre > 0)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _he_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


class _AdamState:
    """Per-parameter Adam moment accumulators."""
    def __init__(self, shapes: list[tuple]):
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.t = 0

    def step(
        self,
        params: list[np.ndarray],
        grads: list[np.ndarray],
        lr: float,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        wd: float = 0.0,
    ) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if wd > 0:
                g = g + wd * p
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g ** 2
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def _build_params(
    rng: np.random.Generator,
    in_dim: int,
    hidden_dims: tuple,
    latent_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Initialise encoder + decoder weight matrices and bias vectors."""
    enc_dims = [in_dim] + list(hidden_dims) + [latent_dim]
    dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [in_dim]
    all_dims = enc_dims + dec_dims[1:]   # share nothing; decoder is separate

    Ws, bs = [], []
    for a, b in zip(all_dims[:-1], all_dims[1:]):
        Ws.append(_he_init(rng, a, b))
        bs.append(np.zeros(b))
    return Ws, bs


def _forward(
    x: np.ndarray,
    Ws: list[np.ndarray],
    bs: list[np.ndarray],
    n_enc: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Full encoder→decoder forward pass.

    Returns (x_hat, z, pre_activations).
    pre_activations[i] is the pre-ReLU value at layer i (used for backprop).
    ReLU is applied at all layers except the latent layer and the output.
    """
    pre = []
    h = x
    for i, (W, b) in enumerate(zip(Ws, bs)):
        a = h @ W + b
        pre.append(a)
        is_latent = (i == n_enc - 1)
        is_output = (i == len(Ws) - 1)
        if is_latent or is_output:
            h = a          # no activation
        else:
            h = _relu(a)
        if is_latent:
            z = h
    x_hat = h
    return x_hat, z, pre


def _encoder_forward(
    x: np.ndarray,
    Ws: list[np.ndarray],
    bs: list[np.ndarray],
    n_enc: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    pre = []
    h = x
    for i in range(n_enc):
        a = h @ Ws[i] + bs[i]
        pre.append(a)
        h = a if (i == n_enc - 1) else _relu(a)
    return h, pre


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def _mse_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 2 * (pred - target) / pred.size


def _ce_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    p = _softmax(logits)
    n = len(labels)
    return float(-np.mean(np.log(p[np.arange(n), labels] + 1e-12)))


def _ce_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p = _softmax(logits)
    p[np.arange(len(labels)), labels] -= 1
    return p / len(labels)


def _backprop(
    x: np.ndarray,
    Ws: list[np.ndarray],
    bs: list[np.ndarray],
    pre: list[np.ndarray],
    n_enc: int,
    dout: np.ndarray,            # gradient w.r.t. x_hat (MSE grad)
    age_dz: np.ndarray | None,   # gradient w.r.t. z from age head (or None)
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Backprop through encoder + decoder; returns dW, db lists."""
    n_layers = len(Ws)
    dWs = [None] * n_layers
    dbs = [None] * n_layers

    # Rebuild activations needed for weight gradients
    acts = [x]
    h = x
    for i, (W, b) in enumerate(zip(Ws, bs)):
        is_latent = (i == n_enc - 1)
        is_output = (i == n_layers - 1)
        h = pre[i] if (is_latent or is_output) else _relu(pre[i])
        acts.append(h)

    # Backprop decoder (layers n_enc .. n_layers-1)
    delta = dout   # d_loss / d_x_hat, shape (batch, in_dim)
    for i in range(n_layers - 1, n_enc - 1, -1):
        dWs[i] = acts[i].T @ delta
        dbs[i] = delta.sum(axis=0)
        d_in = delta @ Ws[i].T
        if i > n_enc:   # apply ReLU backprop for hidden decoder layers
            d_in = _relu_back(d_in, pre[i - 1])
        delta = d_in

    # delta is now d_loss / d_z; add age head gradient if present
    if age_dz is not None:
        delta = delta + age_dz

    # Backprop encoder (layers 0 .. n_enc-1)
    for i in range(n_enc - 1, -1, -1):
        dWs[i] = acts[i].T @ delta
        dbs[i] = delta.sum(axis=0)
        d_in = delta @ Ws[i].T
        if i > 0:
            d_in = _relu_back(d_in, pre[i - 1])
        delta = d_in

    return dWs, dbs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Train a dense autoencoder and return latent vectors for all rows."""
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

    in_dim = X.shape[1]
    latent_dim = int(min(max(2, latent_dim), max(2, in_dim - 1)))
    n_enc = len(hidden_dims) + 1   # number of encoder layers

    Ws, bs = _build_params(rng, in_dim, hidden_dims, latent_dim)
    adam = _AdamState([w.shape for w in Ws] + [b.shape for b in bs])

    X_tr = X[train_idx].astype(np.float64)
    X_val = X[val_idx].astype(np.float64)

    best_val = float("inf")
    best_Ws = [w.copy() for w in Ws]
    best_bs = [b.copy() for b in bs]
    stale = 0
    history = []

    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(X_tr))
        tr_losses = []
        for start in range(0, len(X_tr), batch_size):
            xb = X_tr[perm[start:start + batch_size]]
            x_hat, _, pre = _forward(xb, Ws, bs, n_enc)
            loss = _mse(x_hat, xb)
            dout = _mse_grad(x_hat, xb)
            dWs, dbs = _backprop(xb, Ws, bs, pre, n_enc, dout, None)
            adam.step(Ws + bs, dWs + dbs, lr=learning_rate, wd=weight_decay)
            tr_losses.append(loss)

        x_val_hat, _, _ = _forward(X_val, Ws, bs, n_enc)
        val_loss = _mse(x_val_hat, X_val)
        history.append((ep, float(np.mean(tr_losses)), val_loss))

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_Ws = [w.copy() for w in Ws]
            best_bs = [b.copy() for b in bs]
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Restore best weights and extract latent vectors
    _, Z, _ = _forward(X.astype(np.float64), best_Ws, best_bs, n_enc)

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


def fit_multitask_autoencoder(
    X: np.ndarray,
    y_age: np.ndarray | None = None,
    y_time_bin: np.ndarray | None = None,
    y_genotype: np.ndarray | None = None,
    n_age_classes: int = 4,
    n_time_bins: int = 3,
    n_genotypes: int = 4,
    age_weight: float = 1.0,
    time_weight: float = 0.5,
    genotype_weight: float = 0.5,
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
    Train a multitask autoencoder with up to three supervision heads:

    - Age classification (y_age): integer labels 0..n_age_classes-1, -1 = unknown.
    - Temporal bin (y_time_bin): "moment in video" self-supervised signal,
      integer labels 0..n_time_bins-1, -1 = unknown.
    - Genotype (y_genotype): integer labels 0..n_genotypes-1, -1 = unknown.

    Any head with all-None or all-unknown labels is silently skipped.
    All heads share the Adam optimizer with the main encoder/decoder.

    Returns latent vectors Z (n_samples × latent_dim) and a training info dict.
    """
    if X.ndim != 2 or len(X) < 4:
        return X, {"used_autoencoder": False, "reason": "insufficient_data"}

    def _valid(arr):
        return arr is not None and len(arr) == len(X)

    has_age = _valid(y_age) and age_weight > 0
    has_time = _valid(y_time_bin) and time_weight > 0
    has_geno = _valid(y_genotype) and genotype_weight > 0

    # Infer n_genotypes from the label array if provided
    if has_geno:
        _geno_arr = np.asarray(y_genotype, dtype=np.int64)
        _present = _geno_arr[_geno_arr >= 0]
        if len(_present) == 0:
            has_geno = False
        else:
            n_genotypes = max(n_genotypes, int(_present.max()) + 1)
    if has_time:
        _time_arr = np.asarray(y_time_bin, dtype=np.int64)
        _present_t = _time_arr[_time_arr >= 0]
        if len(_present_t) == 0:
            has_time = False
        else:
            n_time_bins = max(n_time_bins, int(_present_t.max()) + 1)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(X) * val_fraction)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) < 2:
        return X, {"used_autoencoder": False, "reason": "insufficient_train_rows"}

    in_dim = X.shape[1]
    latent_dim = int(min(max(2, latent_dim), max(2, in_dim - 1)))
    n_enc = len(hidden_dims) + 1

    Ws, bs = _build_params(rng, in_dim, hidden_dims, latent_dim)

    # Aux heads — always initialised so Adam indices stay stable; unused heads
    # receive zero gradients and are never updated meaningfully.
    W_age = _he_init(rng, latent_dim, n_age_classes)
    b_age = np.zeros(n_age_classes)
    W_time = _he_init(rng, latent_dim, n_time_bins)
    b_time = np.zeros(n_time_bins)
    W_geno = _he_init(rng, latent_dim, n_genotypes)
    b_geno = np.zeros(n_genotypes)

    all_params = Ws + bs + [W_age, b_age, W_time, b_time, W_geno, b_geno]
    adam = _AdamState([p.shape for p in all_params])

    X_tr = X[train_idx].astype(np.float64)
    X_val = X[val_idx].astype(np.float64)

    age_arr  = np.asarray(y_age,      dtype=np.int64) if has_age  else None
    time_arr = np.asarray(y_time_bin, dtype=np.int64) if has_time else None
    geno_arr = np.asarray(y_genotype, dtype=np.int64) if has_geno else None

    age_tr  = age_arr[train_idx]  if has_age  else None
    time_tr = time_arr[train_idx] if has_time else None
    geno_tr = geno_arr[train_idx] if has_geno else None

    age_val  = age_arr[val_idx]  if has_age  else None
    time_val = time_arr[val_idx] if has_time else None
    geno_val = geno_arr[val_idx] if has_geno else None

    best_val = float("inf")
    best_Ws = [w.copy() for w in Ws]
    best_bs = [b.copy() for b in bs]
    best_W_age,  best_b_age  = W_age.copy(),  b_age.copy()
    best_W_time, best_b_time = W_time.copy(), b_time.copy()
    best_W_geno, best_b_geno = W_geno.copy(), b_geno.copy()
    stale = 0
    history = []

    def _head_loss_and_grads(z, labels, W_head, b_head, weight):
        """CE loss + grads for one classification head. Returns (loss, dz, dW, db)."""
        mask = labels >= 0
        if not mask.any():
            return 0.0, np.zeros_like(z), np.zeros_like(W_head), np.zeros_like(b_head)
        logits = z[mask] @ W_head + b_head
        loss = weight * _ce_loss(logits, labels[mask])
        d_logits = weight * _ce_grad(logits, labels[mask])
        dW = z[mask].T @ d_logits
        db = d_logits.sum(axis=0)
        dz = np.zeros_like(z)
        dz[mask] = d_logits @ W_head.T
        return loss, dz, dW, db

    for ep in range(1, epochs + 1):
        perm = rng.permutation(len(X_tr))
        tr_losses = []
        for start in range(0, len(X_tr), batch_size):
            bi = perm[start:start + batch_size]
            xb = X_tr[bi]
            x_hat, z, pre = _forward(xb, Ws, bs, n_enc)

            loss = _mse(x_hat, xb)
            dout = _mse_grad(x_hat, xb)
            aux_dz = np.zeros_like(z)

            dW_age  = np.zeros_like(W_age)
            db_age_ = np.zeros_like(b_age)
            dW_time  = np.zeros_like(W_time)
            db_time_ = np.zeros_like(b_time)
            dW_geno  = np.zeros_like(W_geno)
            db_geno_ = np.zeros_like(b_geno)

            if has_age:
                l, dz, dW_age, db_age_ = _head_loss_and_grads(
                    z, age_tr[bi], W_age, b_age, age_weight)
                loss += l; aux_dz += dz
            if has_time:
                l, dz, dW_time, db_time_ = _head_loss_and_grads(
                    z, time_tr[bi], W_time, b_time, time_weight)
                loss += l; aux_dz += dz
            if has_geno:
                l, dz, dW_geno, db_geno_ = _head_loss_and_grads(
                    z, geno_tr[bi], W_geno, b_geno, genotype_weight)
                loss += l; aux_dz += dz

            dWs, dbs = _backprop(xb, Ws, bs, pre, n_enc, dout, aux_dz)
            adam.step(
                Ws + bs + [W_age, b_age, W_time, b_time, W_geno, b_geno],
                dWs + dbs + [dW_age, db_age_, dW_time, db_time_, dW_geno, db_geno_],
                lr=learning_rate, wd=weight_decay,
            )
            tr_losses.append(loss)

        # Validation loss
        x_val_hat, z_val, _ = _forward(X_val, Ws, bs, n_enc)
        val_loss = _mse(x_val_hat, X_val)
        if has_age and age_val is not None:
            mask = age_val >= 0
            if mask.any():
                val_loss += age_weight * _ce_loss(z_val[mask] @ W_age + b_age, age_val[mask])
        if has_time and time_val is not None:
            mask = time_val >= 0
            if mask.any():
                val_loss += time_weight * _ce_loss(z_val[mask] @ W_time + b_time, time_val[mask])
        if has_geno and geno_val is not None:
            mask = geno_val >= 0
            if mask.any():
                val_loss += genotype_weight * _ce_loss(z_val[mask] @ W_geno + b_geno, geno_val[mask])

        history.append((ep, float(np.mean(tr_losses)), val_loss))

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_Ws = [w.copy() for w in Ws]
            best_bs = [b.copy() for b in bs]
            best_W_age,  best_b_age  = W_age.copy(),  b_age.copy()
            best_W_time, best_b_time = W_time.copy(), b_time.copy()
            best_W_geno, best_b_geno = W_geno.copy(), b_geno.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    _, Z, _ = _forward(X.astype(np.float64), best_Ws, best_bs, n_enc)

    def _head_acc(Z, labels, W, b):
        if labels is None:
            return None
        mask = labels >= 0
        if not mask.any():
            return None
        preds = (Z[mask] @ W + b).argmax(axis=1)
        return float((preds == labels[mask]).mean())

    age_acc  = _head_acc(Z, age_arr,  best_W_age,  best_b_age)
    time_acc = _head_acc(Z, time_arr, best_W_time, best_b_time)
    geno_acc = _head_acc(Z, geno_arr, best_W_geno, best_b_geno)

    if verbose:
        parts = [
            f"[multitask_ae] epochs={len(history)} latent_dim={latent_dim}",
            f"best_val_loss={best_val:.6g}",
        ]
        if age_acc  is not None: parts.append(f"age_acc={age_acc:.3f}")
        if time_acc is not None: parts.append(f"time_acc={time_acc:.3f}")
        if geno_acc is not None: parts.append(f"geno_acc={geno_acc:.3f}")
        print(" ".join(parts))

    return Z, {
        "used_autoencoder": True,
        "multitask": True,
        "latent_dim": latent_dim,
        "epochs_trained": len(history),
        "best_val_loss": best_val,
        "age_accuracy":      age_acc,
        "time_accuracy":     time_acc,
        "genotype_accuracy": geno_acc,
    }
