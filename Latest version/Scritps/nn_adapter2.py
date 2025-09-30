import torch
import numpy as np
from typing import Dict, Tuple, List, Iterable, Optional, Any

from model2 import PolicyValueNet
# If you put the encoder in a module, prefer importing it:
# from encoder import encode_env as encode_env_torch
# For a self-contained adapter, we keep a small local encoder replica:

PIECES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
Move = Tuple[int, int, int, int]

def _encode_state_np(env) -> np.ndarray:
    """18x8x8 planes; matches your encoder."""
    x = np.zeros((18, 8, 8), dtype=np.float32)
    # pieces
    for r in range(8):
        for c in range(8):
            p = env.board[r][c]
            if p == 0:
                continue
            idx = PIECES[abs(p)] + (0 if p > 0 else 6)
            x[idx, r, c] = 1.0
    # side to move
    x[12, :, :] = 1.0 if env.current_player == 1 else 0.0
    # castling
    cr = getattr(env, "castling_rights", {}) or {}
    x[13, :, :] = 1.0 if cr.get("K", False) else 0.0
    x[14, :, :] = 1.0 if cr.get("Q", False) else 0.0
    x[15, :, :] = 1.0 if cr.get("k", False) else 0.0
    x[16, :, :] = 1.0 if cr.get("q", False) else 0.0
    # en-passant file
    ep = getattr(env, "en_passant_square", None)
    if isinstance(ep, tuple) and len(ep) == 2:
        _, file_c = ep
        if isinstance(file_c, int) and 0 <= file_c < 8:
            x[17, :, file_c] = 1.0
    return x

def _encode_state_torch(env, device: torch.device) -> torch.Tensor:
    arr = _encode_state_np(env)
    t = torch.from_numpy(arr).to(device)
    return t

class PVWrapper:
    """
    Adapter for factored head:
      - net(x) -> from_logits [B,64], delta_logits [B,64,73], value [B,1]
      - policy(env) uses env.masks_factored() and env.move_to_factored(move)
      - value batching hooks for the agent's micro-batching path
    """
    # Optional: tell the agent a good batch size for GPU
    preferred_batch: int = 64

    def __init__(self, net: PolicyValueNet, device: str = "cpu"):
        self.net = net.to(device)
        self.device = torch.device(device)
        self.net.eval()

    # ---------- POLICY ----------
    @torch.no_grad()
    def policy(self, env) -> Dict[Move, float]:
        """
        Return priors over legal moves using the factored head:
          logit(move) = from_logits[from_idx] + delta_logits[from_idx, delta_idx]
        Masks:
          - use env.masks_factored() to zero out illegals
          - also double-check with env.legal_moves() in case caller expects that
        """
        # Encode features
        x = _encode_state_torch(env, self.device).unsqueeze(0)  # [1,18,8,8]

        # Forward
        out = self.net(x)
        if isinstance(out, tuple) and len(out) == 3:
            from_logits, delta_logits, _v = out
        else:
            raise RuntimeError("Policy head mismatch: model must return (from_logits, delta_logits, value).")

        # Shapes
        # from_logits: [1,64], delta_logits: [1,64,73]
        from_logits = from_logits[0]       # [64]
        delta_logits = delta_logits[0]     # [64,73]

        # Masks (numpy uint8) -> torch
        if not hasattr(env, "masks_factored") or not hasattr(env, "move_to_factored"):
            raise RuntimeError("Env must provide masks_factored() and move_to_factored() for the factored head.")
        from_mask_np, delta_mask_np = env.masks_factored()  # [64], [64,73] uint8
        from_mask = torch.from_numpy(from_mask_np.astype(np.bool_)).to(self.device)        # [64]
        delta_mask = torch.from_numpy(delta_mask_np.astype(np.bool_)).to(self.device)      # [64,73]

        # Masked combined logits: L[i,d] = from[i] + delta[i,d], but only where legal
        # We’ll extract only legal moves anyway, but the masked matrix is handy for clarity.
        # Set illegal entries to -inf so they don't contribute if you ever softmax the full grid.
        neg_inf = torch.finfo(from_logits.dtype).min
        # broadcast add
        L = from_logits.view(64, 1) + delta_logits  # [64,73]
        L = torch.where(delta_mask, L, torch.full_like(L, neg_inf))

        # Gather legal moves and their indices via env.legal_moves() -> move_to_factored
        try:
            legal: List[Move] = env.legal_moves()
        except Exception:
            legal = []
        if not legal:
            return {}

        # Build vector of logits for just the legal moves
        legal_logits: List[float] = []
        mapped: List[Tuple[int, int]] = []  # (from_idx, delta_idx) for each legal move
        for mv in legal:
            i, d = env.move_to_factored(mv)  # should be valid given the legal move
            mapped.append((i, d))
            legal_logits.append(float(L[i, d].item()))

        # Softmax over the legal set only (numeric stability)
        logits_arr = np.asarray(legal_logits, dtype=np.float64)
        m = np.max(logits_arr)
        exps = np.exp(logits_arr - m)
        z = exps.sum()
        if not np.isfinite(z) or z <= 0.0:
            p = np.full_like(exps, 1.0 / len(exps))
        else:
            p = exps / z

        # Return dict over concrete board moves
        priors = {mv: float(max(1e-12, pi)) for mv, pi in zip(legal, p)}
        return priors

    # ---------- VALUE ----------
    @torch.no_grad()
    def value(self, env, perspective: int) -> float:
        """
        Scalar value in [-1,1] from the given perspective.
        """
        x = _encode_state_torch(env, self.device).unsqueeze(0)  # [1,18,8,8]
        out = self.net(x)
        if isinstance(out, tuple) and len(out) == 3:
            _from_logits, _delta_logits, v = out
        else:
            # For safety, also accept (p, v) older signatures
            try:
                _p, v = out
            except Exception as _:
                raise RuntimeError("Value head mismatch: model must return (from_logits, delta_logits, value).")
        v = float(v.view(-1).item())
        return v if perspective == 1 else -v

    # ---------- OPTIONAL BATCH HOOKS FOR AGENT ----------
    @torch.no_grad()
    def features_from_env(self, env) -> torch.Tensor:
        """
        Agent will call this for batched value evaluation.
        Returns a *CPU* tensor [18,8,8]; agent will pass a list of these.
        """
        # Keep on CPU; we'll stack and move in value_many_features.
        return torch.from_numpy(_encode_state_np(env))

    @torch.no_grad()
    def value_many_features(self, feats: Iterable[torch.Tensor], perspective: int) -> List[float]:
        """
        Batched value eval from pre-extracted features (list of [18,8,8] CPU tensors).
        """
        if not feats:
            return []
        X = torch.stack([f if isinstance(f, torch.Tensor) else torch.from_numpy(f) for f in feats], dim=0)
        X = X.to(self.device, non_blocking=True)
        out = self.net(X)
        if isinstance(out, tuple) and len(out) == 3:
            _from_logits, _delta_logits, v = out
        else:
            try:
                _p, v = out
            except Exception:
                raise RuntimeError("Value head mismatch in batch: expected (from_logits, delta_logits, value).")
        v = v.view(-1).tolist()
        if perspective == 1:
            return [float(x) for x in v]
        return [float(-x) for x in v]

    @torch.no_grad()
    def value_many(self, envs: Iterable[Any], perspective: int) -> List[float]:
        """
        Fallback batch interface if the agent doesn't use features_from_env.
        """
        feats = [torch.from_numpy(_encode_state_np(e)) for e in envs]
        return self.value_many_features(feats, perspective)
