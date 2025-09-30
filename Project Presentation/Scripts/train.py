import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import PolicyValueNet  # must output (from_logits, delta_logits, value)

# ----------------------------
# Dataset
# ----------------------------
class ChessDataset(Dataset):
    """
    Factored-policy chess dataset loader.

    Expects NPZ files (one or more) with:
      - X  : float32 [N, 18, 8, 8]     input planes
      - PF : float32 [N, 64]           from-square marginal (sum=1 over 64)
      - PD : float32 [N, 64, 73]       per-from conditional over deltas
                                       (for each i with legal moves, PD[i,:] sums to 1)
      - Z  : float32 [N] or [N,1]      scalar value in [-1, 1] (side-to-move outcome)

    Backward-compat: if a file lacks PF/PD (legacy flat 4096 policy), we raise an error
    and ask you to regenerate with factored targets.
    """
    def __init__(self, paths):
        xs, pfs, pds, zs = [], [], [], []
        for p in paths:
            d = np.load(p)
            if "PF" not in d or "PD" not in d:
                raise ValueError(
                    f"{p} is legacy/flat and lacks PF/PD. "
                    f"Regenerate dataset with factored targets (PF:[64], PD:[64,73])."
                )
            xs.append(d["X"].astype(np.float32))
            pfs.append(d["PF"].astype(np.float32))
            pds.append(d["PD"].astype(np.float32))
            z = d["Z"].astype(np.float32)
            # Accept [N] or [N,1] for Z
            if z.ndim == 2 and z.shape[1] == 1:
                z = z[:, 0]
            zs.append(z)

        # Concatenate across all NPZ files
        self.X = np.concatenate(xs, axis=0)
        self.PF = np.concatenate(pfs, axis=0)
        self.PD = np.concatenate(pds, axis=0)
        self.Z = np.concatenate(zs, axis=0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
            X  : torch.float32 [18,8,8]
            PF : torch.float32 [64]
            PD : torch.float32 [64,73]
            Z  : torch.float32 [] (scalar)
        """
        return (
            torch.from_numpy(self.X[idx]),                    # [18,8,8]
            torch.from_numpy(self.PF[idx]),                   # [64]
            torch.from_numpy(self.PD[idx]),                   # [64,73]
            torch.tensor(self.Z[idx], dtype=torch.float32),   # []
        )


# ----------------------------
# Helpers
# ----------------------------
@torch.no_grad()
def _sharpen_probs(p: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Temperature-like transform on simplex distributions along the last dim.

    Input:
        p:    >=0, sums to 1 along last dim (any leading dims ok)
        alpha:
            - alpha == 1    → no change
            - alpha <  1    → FLATTENS (smooths) distribution
            - alpha >  1    → SHARPENS distribution

    Returns:
        q = normalize(p ** alpha) along the last dimension
    """
    if alpha is None or abs(alpha - 1.0) < 1e-8:
        return p
    q = torch.clamp(p, min=1e-12) ** alpha
    q = q / (q.sum(dim=-1, keepdim=True) + 1e-12)
    return q


# ----------------------------
# Training
# ----------------------------
def train_one_epoch(net: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    device: str = "cpu",
                    grad_clip: float = 1.0,
                    ema_model: AveragedModel | None = None,
                    sharpen_alpha: float = 0.8) -> float:
    """
    One epoch of supervised training for a factored policy + value network.

    Policy factorization (per sample):
        Let p_from = softmax(from_logits) in R^{64}
            p_delta[i] = softmax(delta_logits[i, :]) in R^{73}

        Targets: PF in Δ^{64}, PD[i] in Δ^{73} for legal i.
        Loss_from  = CE(PF || p_from) = - Σ_i PF[i] log p_from[i]
        Loss_delta = Σ_i PF[i] * CE(PD[i] || p_delta[i])
        Loss_policy = mean(Loss_from + Loss_delta) over batch

    Value loss:
        Smooth L1 (Huber) between v.squeeze(1) and scalar Z ∈ [-1,1].

    Args:
        grad_clip: L2 max norm for gradient clipping (None/≤0 disables).
        ema_model: SWA/AveragedModel; if provided, updated after each optimizer step.
        sharpen_alpha: passed to `_sharpen_probs`; note alpha<1 FLATTENS targets.

    Returns:
        Average total loss over the entire dataset (sum over batches / N).
    """
    net.train()
    total_loss = 0.0
    huber = nn.SmoothL1Loss(reduction="mean")

    for X, PF, PD, Z in loader:
        # Move to device (pin_memory in DataLoader accelerates H2D on CUDA)
        X  = X.to(device, non_blocking=True)        # [B,18,8,8]
        PF = PF.to(device, non_blocking=True)       # [B,64]
        PD = PD.to(device, non_blocking=True)       # [B,64,73]
        Z  = Z.to(device, non_blocking=True)        # [B]

        # Forward: model must return ([B,64], [B,64,73], [B,1])
        from_logits, delta_logits, v = net(X)
        v = v.squeeze(1)                            # [B]

        # Log-softmax for stable CE with soft targets
        log_p_from  = torch.log_softmax(from_logits, dim=1)  # [B,64]
        log_p_delta = torch.log_softmax(delta_logits, dim=2) # [B,64,73]

        # Optional temperature transform on targets (alpha<1 = smoothing)
        with torch.no_grad():
            PF_tgt = _sharpen_probs(PF, alpha=sharpen_alpha)  # [B,64]
            PD_tgt = _sharpen_probs(PD, alpha=sharpen_alpha)  # [B,64,73]

        # From-square CE: -(PF · log p_from)
        loss_from = -(PF_tgt * log_p_from).sum(dim=1)         # [B]

        # Row-wise CE over deltas, then weight by PF via law of total probability
        ce_rows = -(PD_tgt * log_p_delta).sum(dim=2)          # [B,64]
        loss_delta = (PF_tgt * ce_rows).sum(dim=1)            # [B]

        loss_policy = (loss_from + loss_delta).mean()

        # Value loss (Huber)
        loss_value = huber(v, Z)

        loss = loss_policy + loss_value

        # Backprop + step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        optimizer.step()

        # EMA update (recommended for evaluation/export)
        if ema_model is not None:
            ema_model.update_parameters(net)

        total_loss += float(loss.item()) * X.size(0)

    # Average over dataset size (not batches)
    return total_loss / len(loader.dataset)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model (must return (from_logits, delta_logits, value))
    net = PolicyValueNet().to(device)

    # Optimizer: AdamW is a solid default for conv + linear stacks
    base_lr = 3e-4
    opt = optim.AdamW(net.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=1e-4)

    # EMA of parameters; behaves like an exponential moving average
    ema_decay = 0.999
    ema_model = AveragedModel(net, avg_fn=lambda avg, p, n: avg * ema_decay + p * (1.0 - ema_decay))

    # Data
    dataset = ChessDataset(["./saved_games_pgn/data/dataset_explore_8.npz"])
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,                      # set >0 for throughput if disk/CPU allow
        pin_memory=(device == "cuda"),
        persistent_workers=False
    )

    # Scheduler: cosine over epochs with warmup
    # NOTE: manual warmup below overrides scheduler each epoch; see notes above.
    epochs = 10
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def warmup_factor(e: int, warmup_epochs: int = 2) -> float:
        # Linear warmup from 0→1 over `warmup_epochs` epochs
        return float(min(1.0, (e + 1) / max(1, warmup_epochs)))

    for epoch in range(epochs):
        # Epoch-wise warmup: directly set optimizer LR (overrides scheduler value)
        wf = warmup_factor(epoch, warmup_epochs=2)
        for g in opt.param_groups:
            g["lr"] = base_lr * wf

        loss = train_one_epoch(
            net,
            loader,
            opt,
            device=device,
            grad_clip=1.0,
            ema_model=ema_model,
            sharpen_alpha=0.8
        )
        print(f"Epoch {epoch+1:02d} | loss={loss:.4f} | lr={opt.param_groups[0]['lr']:.6f}")

        # This updates scheduler state, but its lr gets overwritten at the next epoch start.
        scheduler.step()

    # Save EMA weights (recommended for inference) and raw weights
    torch.save(ema_model.module.state_dict(), "policy_value_net.pt")
    torch.save(net.state_dict(), "policy_value_net_raw.pt")
