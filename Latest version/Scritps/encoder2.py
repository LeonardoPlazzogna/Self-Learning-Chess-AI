import torch
import numpy as np
from typing import Iterable, Optional, Tuple

# Mapping used by your env: + = white, - = black
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
    Rotate an [C,8,8] feature tensor 180° (flip both axes).
    Useful if you want to canonicalize features so side-to-move is 'white'.
    Not used by default.
    """
    return np.flip(np.flip(x, axis=-1), axis=-2).copy()

def encode_env_np(env, canonical_to_white: bool = False) -> np.ndarray:
    """
    Numpy encoder.
    Returns float32 array of shape [18, 8, 8] with {0,1} values.

    Plane order:
      0..5:   White P,N,B,R,Q,K
      6..11:  Black P,N,B,R,Q,K
      12:     side to move (all-ones if white to move else zeros)
      13..16: castling rights (wK,wQ,bK,bQ) replicated on board
      17:     en-passant file (one-hot column replicated on ranks) else zeros

    Args
    ----
    env : your ChessEnv (expects .board, .current_player, .castling_rights, .en_passant_square)
    canonical_to_white : if True, rotate features so that side-to-move is always 'white'.
                         Default False; leave as False to match your factored policy mapping.
    """
    x = np.zeros((18, 8, 8), dtype=np.float32)

    # Pieces
    board = getattr(env, "board", None)
    if board is None or len(board) != 8 or any(len(row) != 8 for row in board):
        raise ValueError("encode_env_np: env.board must be an 8x8 list of ints.")
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == 0:
                continue
            idx = PIECES.get(abs(p))
            if idx is None:
                raise ValueError(f"encode_env_np: unknown piece code {p} at ({r},{c}).")
            idx += (0 if p > 0 else 6)  # white planes 0..5, black planes 6..11
            x[idx, r, c] = 1.0

    # Side to move
    stm = int(getattr(env, "current_player", 1))
    x[12, :, :] = 1.0 if stm == 1 else 0.0

    # Castling rights dictionary: {"K","Q","k","q"}
    cr = getattr(env, "castling_rights", {}) or {}
    wk = bool(cr.get("K", False))
    wq = bool(cr.get("Q", False))
    bk = bool(cr.get("k", False))
    bq = bool(cr.get("q", False))
    x[13, :, :] = 1.0 if wk else 0.0
    x[14, :, :] = 1.0 if wq else 0.0
    x[15, :, :] = 1.0 if bk else 0.0
    x[16, :, :] = 1.0 if bq else 0.0

    # En-passant file (column replicated)
    ep = getattr(env, "en_passant_square", None)
    if isinstance(ep, tuple) and len(ep) == 2:
        _, file_c = ep
        if isinstance(file_c, int) and 0 <= file_c < 8:
            x[17, :, file_c] = 1.0

    # Optional canonicalization (NOT needed for your factored delta mapping)
    if canonical_to_white and stm == -1:
        x = _rotate180_board_like(x)
        # Side-to-move plane remains as "white to move" semantics after rotation:
        # we want plane 12 = ones since we've canonicalized so STM is 'white'.
        x[12, :, :] = 1.0

        # Castling planes are board-aligned hints; after rotation, swap white/black planes.
        # Planes: 13=wK, 14=wQ, 15=bK, 16=bQ
        x[[13, 14, 15, 16]] = x[[15, 16, 13, 14]]

    return x

def encode_env(env, canonical_to_white: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Torch encoder (single env).
    Returns torch.float32 tensor of shape [18, 8, 8] on `device` (if provided).
    """
    x = encode_env_np(env, canonical_to_white=canonical_to_white)
    t = torch.from_numpy(x)
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
        [B, 18, 8, 8] float32 tensor
    """
    xs = [encode_env_np(e, canonical_to_white=canonical_to_white) for e in envs]
    arr = np.stack(xs, axis=0)  # [B, 18, 8, 8]
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t

def encode_env_with_masks(env,
                          canonical_to_white: bool = False,
                          device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Convenience for the factored head path:
    Returns (features, from_mask, delta_mask) where:
        features  : [18, 8, 8]
        from_mask : [64] uint8 or None if env has no masks_factored()
        delta_mask: [64, 73] uint8 or None if env has no masks_factored()

    NOTE: The masks are *not* rotated here; your env's move mapping (move_to_factored)
          already canonicalizes deltas by side-to-move, and the agent/adapter should
          fetch masks directly from env when needed. This helper is just ergonomic.
    """
    feat = encode_env(env, canonical_to_white=canonical_to_white, device=device)

    from_mask_t = None
    delta_mask_t = None
    if hasattr(env, "masks_factored"):
        try:
            from_mask, delta_mask = env.masks_factored()  # numpy uint8 as implemented in your env
            from_mask_t = torch.from_numpy(from_mask.astype(np.uint8))
            delta_mask_t = torch.from_numpy(delta_mask.astype(np.uint8))
            if device is not None:
                from_mask_t = from_mask_t.to(device)
                delta_mask_t = delta_mask_t.to(device)
        except Exception:
            # If mask construction fails for any reason, fall back to features only.
            from_mask_t = None
            delta_mask_t = None

    return feat, from_mask_t, delta_mask_t
