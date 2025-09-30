import torch
import numpy as np
from typing import Dict, Tuple, List, Iterable, Optional, Any

from model import PolicyValueNet

# Map absolute piece IDs → plane indices (0..5 for white, 6..11 for black)
# Convention: env.board[r][c] holds integers: +1..+6 (white), -1..-6 (black), 0 empty
PIECES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# A move is represented by (r_from, c_from, r_to, c_to)
Move = Tuple[int, int, int, int]


def _encode_state_np(env) -> np.ndarray:
    """
    Build 18×8×8 input planes from `env` (numpy, float32).

    Planes (index → meaning):
      0..5   : white piece onehots (P,N,B,R,Q,K)
      6..11  : black piece onehots (p,n,b,r,q,k)
      12     : side to move (all-ones if current_player==+1 else zeros)
      13..16 : castling rights (K,Q,k,q) broadcast over 8×8
      17     : en-passant file (1s on that file for all ranks)

    Assumptions:
      - env.board is 8×8 with integers in {0, ±1..±6}.
      - env.current_player ∈ {+1, -1}.
      - env.castling_rights is a dict-like with keys 'K','Q','k','q'→bool.
      - env.en_passant_square is (rank, file) or None.
    """
    x = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes
    for r in range(8):
        for c in range(8):
            p = env.board[r][c]
            if p == 0:
                continue
            idx = PIECES.get(abs(p), None)
            if idx is None:
                # Unknown piece ID: skip silently or raise depending on your env policy
                continue
            # White in 0..5, Black in 6..11
            plane = idx + (0 if p > 0 else 6)
            x[plane, r, c] = 1.0

    # Side-to-move (broadcast)
    x[12, :, :] = 1.0 if getattr(env, "current_player", 1) == 1 else 0.0

    # Castling rights (broadcast)
    cr = getattr(env, "castling_rights", {}) or {}
    x[13, :, :] = 1.0 if cr.get("K", False) else 0.0
    x[14, :, :] = 1.0 if cr.get("Q", False) else 0.0
    x[15, :, :] = 1.0 if cr.get("k", False) else 0.0
    x[16, :, :] = 1.0 if cr.get("q", False) else 0.0

    # En-passant file (if any)
    ep = getattr(env, "en_passant_square", None)
    if isinstance(ep, tuple) and len(ep) == 2:
        _, file_c = ep
        if isinstance(file_c, int) and 0 <= file_c < 8:
            x[17, :, file_c] = 1.0

    return x


def _encode_state_torch(env, device: torch.device) -> torch.Tensor:
    """
    Torch wrapper for encoder. Returns FloatTensor [18,8,8] on `device`.
    """
    arr = _encode_state_np(env)
    t = torch.from_numpy(arr).to(device)
    return t


class PVWrapper:
    """
    Lightweight adapter around a factored-head Policy-Value net.

    Contract with the model:
        net(x) -> from_logits [B,64], delta_logits [B,64,73], value [B,1]

    Contract with the env (must provide):
        - masks_factored() -> (from_mask [64], delta_mask [64,73]) as numpy uint8/bool
        - move_to_factored(move) -> (from_idx:int in [0,64), delta_idx:int in [0,73))
        - legal_moves() -> List[Move]
        - board/current_player/castling_rights/en_passant_square for encoding

    Methods:
        - policy(env) : Dict[Move,float] priors over legal moves (softmax on legal set)
        - value(env, perspective) : float in [-1,1]; flipped if perspective == -1
        - features_from_env / value_many_features / value_many : batched helpers
    """

    # Hint to an upper-level agent about a good batch size for GPU evals
    preferred_batch: int = 64

    def __init__(self, net: PolicyValueNet, device: str = "cpu"):
        self.net = net.to(device)
        self.device = torch.device(device)
        self.net.eval()  # inference mode (still computes grads off due to no_grad decorators)
    
    @torch.no_grad()
    def __call__(self, env, perspective: int) -> float:
        # Allow passing the whole adapter as value_fn
        return self.value(env, perspective)

    # ---------- POLICY ----------
    @torch.no_grad()
    def policy_factored(self, env, from_mask, delta_mask):
        """
        Return raw logits for the agent's factored path.
        Shapes: from_logits [64], delta_logits [64,73] (numpy float32).
        """
        x = _encode_state_torch(env, self.device).unsqueeze(0)
        out = self.net(x)
        if not (isinstance(out, tuple) and len(out) == 3):
            raise RuntimeError("Model must return (from_logits, delta_logits, value).")
        from_logits, delta_logits, _ = out
        return (from_logits[0].detach().cpu().numpy().astype(np.float32),
                delta_logits[0].detach().cpu().numpy().astype(np.float32))
    
    @torch.no_grad()
    def policy(self, env) -> Dict[Move, float]:
        """
        Compute prior probabilities over legal moves using the factored policy.

        Combined logit for a move (from_idx i, delta_idx d):
            L[i,d] = from_logits[i] + delta_logits[i,d]

        We:
          1) encode the state,
          2) run the net to get logits,
          3) apply legality masks,
          4) gather logits for the *legal* move list,
          5) do a numerically-stable softmax over the legal set only.
        """
        # Encode to [1,18,8,8]
        x = _encode_state_torch(env, self.device).unsqueeze(0)

        # Forward
        out = self.net(x)
        if isinstance(out, tuple) and len(out) == 3:
            from_logits, delta_logits, _v = out
        else:
            raise RuntimeError("Policy head mismatch: model must return (from_logits, delta_logits, value).")

        # Shapes after squeeze: from_logits [64], delta_logits [64,73]
        from_logits = from_logits[0]
        delta_logits = delta_logits[0]

        # Masks (numpy -> torch.bool on the same device)
        if not hasattr(env, "masks_factored") or not hasattr(env, "move_to_factored"):
            raise RuntimeError("Env must provide masks_factored() and move_to_factored() for the factored head.")
        from_mask_np, delta_mask_np = env.masks_factored()  # [64], [64,73]
        from_mask = torch.from_numpy(from_mask_np.astype(np.bool_)).to(self.device)
        delta_mask = torch.from_numpy(delta_mask_np.astype(np.bool_)).to(self.device)

        # Build full grid of combined logits: L[i,d] = from[i] + delta[i,d]
        L = from_logits.view(64, 1) + delta_logits  # [64,73]

        # Mask illegals to -inf so a full-grid softmax (if used) ignores them.
        # NOTE: delta_mask typically already encodes from-legality row-wise.
        neg_inf = torch.finfo(L.dtype).min
        L = torch.where(delta_mask, L, torch.full_like(L, neg_inf))
        # Optional extra guard (uncomment if your delta_mask doesn't encode from-mask):
        # L = torch.where(from_mask.view(64,1), L, torch.full_like(L, neg_inf))

        # Legal moves from env; fall back to empty set on exception
        try:
            legal: List[Move] = env.legal_moves()
        except Exception:
            legal = []
        if not legal:
            return {}

        # Collect logits for the legal list (uses env.move_to_factored mapping)
        legal_logits: List[float] = []
        mapped: List[Tuple[int, int]] = []  # (from_idx, delta_idx) per legal move
        for mv in legal:
            i, d = env.move_to_factored(mv)
            mapped.append((i, d))
            legal_logits.append(float(L[i, d].item()))

        # Numerically stable softmax over *legal* moves only
        logits_arr = np.asarray(legal_logits, dtype=np.float64)
        m = np.max(logits_arr)
        exps = np.exp(logits_arr - m)
        z = exps.sum()
        if not np.isfinite(z) or z <= 0.0:
            # Degenerate case: default to uniform over legal moves
            p = np.full_like(exps, 1.0 / len(exps))
        else:
            p = exps / z

        # Dict[Move, float] with small floor to avoid exact zeros downstream
        priors = {mv: float(max(1e-12, pi)) for mv, pi in zip(legal, p)}
        return priors

    # ---------- VALUE ----------
    @torch.no_grad()
    def value(self, env, perspective: int) -> float:
        """
        Evaluate scalar value in [-1,1] for the given perspective.
        If perspective == +1, return v; if -1, return -v.
        """
        x = _encode_state_torch(env, self.device).unsqueeze(0)  # [1,18,8,8]
        out = self.net(x)
        if isinstance(out, tuple) and len(out) == 3:
            _from_logits, _delta_logits, v = out
        else:
            # Back-compat acceptance of (p, v) signatures
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
        Return CPU tensor [18,8,8] for micro-batching upstream.
        Keeping features on CPU lets the caller stack many and DMA once.
        """
        return torch.from_numpy(_encode_state_np(env))

    @torch.no_grad()
    def value_many_features(self, feats: Iterable[torch.Tensor], perspective: int) -> List[float]:
        """
        Batched value evaluation from pre-extracted features.
        `feats`: iterable of CPU tensors/arrays shaped [18,8,8].
        """
        feats = list(feats)
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
        vals = v.view(-1).tolist()
        return [float(x if perspective == 1 else -x) for x in vals]

    @torch.no_grad()
    def value_many(self, envs: Iterable[Any], perspective: int) -> List[float]:
        """
        Convenience path if the agent doesn't pre-extract features.
        Encodes each env, then defers to `value_many_features`.
        """
        feats = [torch.from_numpy(_encode_state_np(e)) for e in envs]
        return self.value_many_features(feats, perspective)
