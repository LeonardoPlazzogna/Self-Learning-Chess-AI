import torch
import numpy as np
from typing import Iterable, Optional, Tuple

# Mapping used by your env: + = white, - = black (±1..±6)
# Planes: 0..5 = white P,N,B,R,Q,K ; 6..11 = black P,N,B,R,Q,K
PIECES = {
    1: 0,  # P
    2: 1,  # N
    3: 2,  # B
    4: 3,  # R
    5: 4,  # Q
    6: 5,  # K
}

def _rotate180_board_like(x: np.ndarray) -> np.ndarray:
    """
    Rotate an [C,8,8] tensor 180° (flip ranks and files).
    Useful for canonicalizing so side-to-move is 'white'.
    NOTE: This rotates *all* planes; callers must fix any planes whose semantics
    change under rotation (e.g., swap castling hints, overwrite STM plane).
    """
    return np.flip(np.flip(x, axis=-1), axis=-2).copy()

def encode_env_np(env, canonical_to_white: bool = False) -> np.ndarray:
    """
    Numpy encoder → [18, 8, 8] float32 with {0,1} values.

    Plane order:
      0..5   : White P,N,B,R,Q,K (one-hot on squares)
      6..11  : Black P,N,B,R,Q,K (one-hot on squares)
      12     : Side to move (all-ones if white to move else zeros)
      13..16 : Castling rights (wK, wQ, bK, bQ), broadcast over board
      17     : En-passant file (one-hot column broadcast over ranks)

    Args
    ----
    env : object exposing
        - board: 8x8 ints in {0, ±1..±6}
        - current_player: +1 (white) or -1 (black)
        - castling_rights: dict-like with keys 'K','Q','k','q' → bool
        - en_passant_square: (rank, file) or None
    canonical_to_white : if True and black to move, rotate 180° so the side
        to move is treated as 'white'; adjusts STM and swaps castling planes.
        Leave False when using factored move/delta mappings tied to absolute squares.
    """
    x = np.zeros((18, 8, 8), dtype=np.float32)

    # --- Pieces ---
    board = getattr(env, "board", None)
    if board is None or len(board) != 8 or any(len(row) != 8 for row in board):
        raise ValueError("encode_env_np: env.board must be an 8x8 list/array of ints.")

    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == 0:
                continue
            idx = PIECES.get(abs(p))
            if idx is None:
                raise ValueError(f"encode_env_np: unknown piece code {p} at ({r},{c}).")
            plane = idx + (0 if p > 0 else 6)  # white 0..5, black 6..11
            x[plane, r, c] = 1.0

    # --- Side to move ---
    stm = int(getattr(env, "current_player", 1))
    x[12, :, :] = 1.0 if stm == 1 else 0.0

    # --- Castling rights (broadcast planes) ---
    cr = getattr(env, "castling_rights", {}) or {}
    x[13, :, :] = 1.0 if bool(cr.get("K", False)) else 0.0
    x[14, :, :] = 1.0 if bool(cr.get("Q", False)) else 0.0
    x[15, :, :] = 1.0 if bool(cr.get("k", False)) else 0.0
    x[16, :, :] = 1.0 if bool(cr.get("q", False)) else 0.0

    # --- En-passant file (column broadcast) ---
    ep = getattr(env, "en_passant_square", None)
    if isinstance(ep, tuple) and len(ep) == 2:
        _, file_c = ep
        if isinstance(file_c, int) and 0 <= file_c < 8:
            x[17, :, file_c] = 1.0

    # --- Optional canonicalization ---
    # Not needed (and typically *undesired*) for your factored-delta mapping.
    if canonical_to_white and stm == -1:
        x = _rotate180_board_like(x)
        # After rotation, we canonicalize STM to 'white to move':
        x[12, :, :] = 1.0
        # Swap castling planes because their semantics are color-specific:
        # planes: 13=wK, 14=wQ, 15=bK, 16=bQ
        x[[13, 14, 15, 16]] = x[[15, 16, 13, 14]]

    return x

def encode_env(env, canonical_to_white: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Torch encoder (single env) → [18, 8, 8] float32 on `device` (if provided).
    """
    x = encode_env_np(env, canonical_to_white=canonical_to_white)
    t = torch.from_numpy(x)  # preserves float32 dtype
    if device is not None:
        t = t.to(device)
    return t

def encode_many(envs: Iterable, canonical_to_white: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Batch encoder.

    Args:
        envs: iterable of envs
        canonical_to_white: see encode_env_np
        device: optional torch device

    Returns:
        Tensor [B, 18, 8, 8] float32
    """
    xs = [encode_env_np(e, canonical_to_white=canonical_to_white) for e in envs]
    arr = np.stack(xs, axis=0)         # [B, 18, 8, 8]
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t

def encode_env_with_masks(
    env,
    canonical_to_white: bool = False,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Convenience for factored-head pipelines.

    Returns:
        features   : [18, 8, 8] float32 tensor
        from_mask  : [64]  uint8 tensor or None if env lacks masks_factored()
        delta_mask : [64, 73] uint8 tensor or None

    NOTE:
      - Masks are returned as provided by env (uint8). Many PyTorch ops prefer bool;
        cast as needed downstream.
      - Masks are *not* rotated here. Your env's move mapping (move_to_factored)
        already canonicalizes deltas by side-to-move; fetch masks directly from env
        when needed to keep indices consistent.
    """
    feat = encode_env(env, canonical_to_white=canonical_to_white, device=device)

    from_mask_t: Optional[torch.Tensor] = None
    delta_mask_t: Optional[torch.Tensor] = None

    if hasattr(env, "masks_factored"):
        try:
            from_mask, delta_mask = env.masks_factored()  # numpy uint8 as per env
            from_mask_t = torch.from_numpy(from_mask.astype(np.uint8))
            delta_mask_t = torch.from_numpy(delta_mask.astype(np.uint8))
            if device is not None:
                from_mask_t = from_mask_t.to(device)
                delta_mask_t = delta_mask_t.to(device)
        except Exception:
            # If mask construction fails, fall back to features-only.
            from_mask_t = None
            delta_mask_t = None

    return feat, from_mask_t, delta_mask_t
