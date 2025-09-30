# sim_selfplay.py
# ============================================================
# Self-play + Training + Arena Promotion Loop
# A: EXPLORE  (policy-only + MCTS; fast push/pop heuristic)
# B: TRAIN    (fit small policy/value net on (X, PI, Z) from EXPLORE)
# C: ARENA    (eval net vs exploration backend; SPRT promotion)
# D: EVALUATE (optional PGNs/datasets with current eval net)
#
# Outputs:
# - PGNs for EXPLORE / EVAL / ARENA (per-game + per-cycle combined)
# - Datasets as .npz (sparse policy labels: PI_IDX/PI_VAL/PI_PTR)
# - CSVs: arena_results.csv (per-cycle), arena_details_c{cycle}.csv (per-game)
# - JSONL: games.jsonl (robust per-game log lines)
# - run manifest per cycle (JSON)
#
# SPEED / QUALITY (this version):
# - NO deepcopy hot-paths: push/pop in heuristics, mate check, mobility
# - Central inference batching + LRU result cache + in-batch de-dup
# - Sparse policies (idx/val/ptr) -> 10–30x smaller than dense 4096
# - Atomic dataset writes (Windows-safe); robust concat of ragged sparse arrays
# - SPRT in arena (H0: p=0.50 vs H1: p=0.55), early stop; also cap by ARENA_GAMES
# - Dirichlet noise anneals over cycles for EXPLORE
# - Train: cosine LR + warmup + grad-clip + EMA; KL = standard xent
# - Resign/adjudication: end hopeless games earlier
#
# NOTE:
# - Expects: env2.ChessEnv and agent4.{MCTSAgent, SearchConfig}
# - Model: model.PolicyValueNet: (B,18,8,8) -> (logits[4096], v[1])
# ============================================================

from typing import Dict, Tuple, List, Optional
import os, sys, math, random, datetime, time, json, csv, io
import numpy as np
from tqdm import tqdm

# Limit CPU thread pools (prevents oversubscription in data workers)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Import environment and MCTS agent
from env3 import ChessEnv
from agent5 import MCTSAgent, SearchConfig

# ---------- Windows / notebook safety ----------
import multiprocessing as mp
import queue as _queue

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and hasattr(ip, "kernel")
    except Exception:
        return False

IS_WIN = os.name == "nt"
IS_NOTEBOOK = _in_notebook()

# -------------------------- CONFIG --------------------------
BASE_SEED = 42
CYCLES = 32
CURRENT_CYCLE: int = 0

# Net toggles
USE_NET_EXPLORE = False
USE_NET_EVAL    = False  # flips True automatically if TRAIN succeeds

# --- NEW: separate candidate (freshly trained) and baseline (exploration/opponent) ---
CANDIDATE_WEIGHTS_PATH = "./policy_value_net.pt"   # training writes here
EXPLORE_WEIGHTS_PATH   = "./explore_baseline.pt"   # updated only on promotion
# Back-compat alias (optional; safe to keep)
NET_WEIGHTS_PATH = CANDIDATE_WEIGHTS_PATH
NET_DEVICE = "cpu"

# ---- Centralized inference (batched cross-process NN evals) ----
CENTRAL_INFERENCE     = True
CENTRAL_MAX_BATCH     = 128
CENTRAL_FLUSH_MS      = 6
NET_FP16              = True

# Game counts
EXPLORE_GAMES = 1000
EVAL_GAMES    = 4
MAX_PLIES     = 180

# Exploration mix (ramped by cycle)
EXPLORE_MCTS_FRACTION_BASE = 0.35

# Policy sampling params (annealed over plies in EXPLORE)
_TAU_POINTS = [(1,1.20),(4,1.00),(8,0.85),(12,0.75),(16,0.70),(20,0.65)]
def tau_schedule_explore(ply: int) -> float:
    turn = ply + 1  # graph is 1-indexed
    if turn <= _TAU_POINTS[0][0]:
        return _TAU_POINTS[0][1]
    for (x0, y0), (x1, y1) in zip(_TAU_POINTS, _TAU_POINTS[1:]):
        if turn <= x1:
            t = (turn - x0) / float(x1 - x0)
            return y0 + t * (y1 - y0)
    return _TAU_POINTS[-1][1]

EXPLORE_POLICY_EPS = 0.01
# EXPLORE root Dirichlet schedule (matches the figure):
# hold cycles 1–3 at high noise; linearly decay to cycle 12; flat afterward.
DIRI_ALPHA_START = 0.30
DIRI_ALPHA_END   = 0.10
DIRI_EPS_START   = 0.25
DIRI_EPS_END     = 0.10
DIRI_HOLD_CYCLES = 3
DIRI_DECAY_END   = 16

def explore_dirichlet_schedule(cycle: int) -> Tuple[float, float]:
    if cycle <= DIRI_HOLD_CYCLES:
        return DIRI_ALPHA_START, DIRI_EPS_START
    if cycle >= DIRI_DECAY_END:
        return DIRI_ALPHA_END, DIRI_EPS_END
    # linear interpolation from (hold+1) .. DECAY_END
    t = (cycle - DIRI_HOLD_CYCLES) / float(DIRI_DECAY_END - DIRI_HOLD_CYCLES)  # in (0,1]
    alpha = DIRI_ALPHA_START + t * (DIRI_ALPHA_END - DIRI_ALPHA_START)
    eps   = DIRI_EPS_START   + t * (DIRI_EPS_END   - DIRI_EPS_START)
    return float(alpha), float(eps)

# Opening hybrid plies
OPEN_POLICY_PLIES_EXPLORE = 10
OPEN_POLICY_PLIES_EVAL    = 0

# --- Opening book controls (EXPLORE only) ---
USE_OPENING_BOOK: bool = True
BOOK_PLIES: int = 8           # number of half-moves we try to guide
BOOK_PROB: float = 0.7        # chance to use a book on a given game

# -------------------- Arena speed knobs --------------------
OPEN_POLICY_PLIES_ARENA   = 0      # CHANGED: was 20
ARENA_MAX_PLIES           = 160
ARENA_QUICKDRAW_HALFMOVES = 100    # CHANGED: was 60

# -------------------- Quickdraw (EXPLORE/EVAL) --------------------
EXPLORE_QUICKDRAW_HALFMOVES = 80
EVAL_QUICKDRAW_HALFMOVES    = 80

# -------------------- Arena parallelism --------------------
ARENA_PARALLEL     = (not IS_NOTEBOOK) and (CENTRAL_INFERENCE or str(NET_DEVICE).lower() == "cpu")
ARENA_MAX_WORKERS  = max(1, (os.cpu_count() or 2) - 1)

# Parallelization (EXPLORE)
PARALLEL_EXPLORE = True and (not IS_NOTEBOOK)
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Per-game progress in parallel EXPLORE
PER_GAME_PROGRESS = False

# Output dirs
OUT_DIR       = "./saved_games_pgn4"
EXPLORE_DIR   = os.path.join(OUT_DIR, "explore")
EVAL_DIR      = os.path.join(OUT_DIR, "eval")
ARENA_DIR     = os.path.join(OUT_DIR, "arena")
DATA_DIR      = os.path.join(OUT_DIR, "data")
LOG_DIR       = os.path.join(OUT_DIR, "logs")
os.makedirs(EXPLORE_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(ARENA_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# JSONL per-game log
GAMES_JSONL = os.path.join(OUT_DIR, "games.jsonl")

# Save datasets
SAVE_EXPLORE_DATASET = True
SAVE_EVAL_DATASET    = True

# Save per-game PGNs (set False to only keep combined PGNs)
WRITE_PER_GAME_PGNS = False

# Fast IO for EXPLORE shards
FAST_IO = True  # kept for compatibility; shards are saved uncompressed now.

# -------------------- Training config --------------------
TRAIN_AFTER_EXPLORE = True
EPOCHS       = 5
BATCH_SIZE   = 512
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VALUE_LOSS_WEIGHT = 1.0
GRAD_CLIP_NORM = 1.0
WARMUP_FRAC = 0.08  # 5% warmup
EMA_DECAY = 0.999

# Cap how many (state,π,z) we keep per game to avoid long shuffles dominating
MAX_SAMPLES_PER_GAME = 192

# >>> NEW: small replay buffer for training
REPLAY_K = 8

# -------------------- Draw penalty / label smoothing --------------------
def get_draw_penalty(cycle: int) -> float:
    return 0.12 if cycle <= 10 else 0.05

DRAW_PENALTY = get_draw_penalty(CURRENT_CYCLE)  # initialize for import-time use
POLICY_SMOOTH_EPS = 0.03  # label smoothing over legal moves

# --- NEW: augmentation / prioritization toggles ---
AUGMENT_SYMMETRY = False           # (disabled here; enabling requires remapping PF/PD)
COLOR_FLIP_AUG   = False           # (disabled here; enabling requires remapping PF/PD)
PRIORITIZED_LOSS = True            # keep on; we'll wire it into _train_on_npz

# -------------------- Resign / adjudication --------------------
RESIGN_AFTER_PLIES = 24
RESIGN_VALUE = -0.90
RESIGN_CONSEC_PLY = 8

# -------------------- SPRT config --------------------
SPRT_USE = True
SPRT_P0 = 0.50
SPRT_P1 = 0.55
SPRT_ALPHA = 0.05
SPRT_BETA  = 0.05
ARENA_GAMES = 75

MINIARENA_GAMES = 20

# -------------------- Time-bank search (simulation budget) --------------------
TB_ENABLE              = True          # master switch
TB_TOTAL_BANK          = 65_000        # total sims available per game (bank)
TB_NOMINAL_PER_MOVE    = 220           # "salary" added to bank each ply, also baseline target
TB_MIN_SIMS_PER_MOVE   = 64            # hard floor per MCTS move
TB_MAX_SIMS_PER_MOVE   = 640           # hard ceiling per MCTS move
TB_LOW_BANK_THRESHOLD  = 6_000         # below this, tighten budgets
TB_OPENING_PLIES       = 12            # mild saving early
TB_ENDGAME_HMC         = 28            # hmc >= this => endgame-ish saving/spending changes
TB_BRANCH_REF          = 28            # reference branching; > raises spend, < saves
TB_BRANCH_MAX_MULT     = 1.8           # cap for branching multiplier
TB_OPENING_MULT        = 0.85          # opening multiplier
TB_MIDDLEGAME_MULT     = 1.00          # middlegame multiplier
TB_ENDGAME_MULT        = 0.95          # endgame multiplier
TB_TIME_TROUBLE_MULT   = 0.75          # if bank is low, tighten budgets
TB_SHARP_BONUS         = 1.20          # if position looks sharp (in-check or big policy entropy), boost a bit

# -------------------- MCTS configs --------------------
CFG = SearchConfig(
    n_simulations=300,
    c_puct=1.25,
    max_depth=48,
    pw_c=3.0,
    pw_alpha=0.5,
    topk_expand=16,

    policy_temperature=1.0,
    root_dirichlet_alpha=0.08,
    root_dirichlet_eps=0.00,

    early_stop_enabled=True,
    early_stop_check_every=16,
    early_stop_min_visits=110,
    early_stop_best_ratio=0.68,

    dynamic_budget_enabled=True,
    dynamic_min_sims=64,
    dynamic_sharp_moves_threshold=12,
    dynamic_wide_moves_threshold=40,
    dynamic_sharp_scale=1.6,
    dynamic_wide_scale=0.7,
    dynamic_opening_scale=0.9,
    repeat_penalty_at_root= 0.10
)

CFG_EXPLORE = SearchConfig(
    n_simulations=256,
    c_puct=1.25,
    max_depth=40,
    pw_c=3.0,
    pw_alpha=0.5,
    topk_expand=16,

    policy_temperature=1.0,
    root_dirichlet_alpha=0.30,
    root_dirichlet_eps=0.25,

    early_stop_enabled=True,
    early_stop_check_every=10,
    early_stop_min_visits=80,
    early_stop_best_ratio=0.70,

    dynamic_budget_enabled=True,
    dynamic_min_sims=64,
    dynamic_sharp_moves_threshold=12,
    dynamic_wide_moves_threshold=40,
    dynamic_sharp_scale=1.5,
    dynamic_wide_scale=0.5,
    dynamic_opening_scale=1.0,
    repeat_penalty_at_root= 0.15
)

CFG_ARENA = SearchConfig(
    n_simulations=240,
    c_puct=1.25,
    max_depth=36,
    pw_c=3.0,
    pw_alpha=0.5,
    topk_expand=16,

    policy_temperature=1.0,
    root_dirichlet_alpha=0.08,
    root_dirichlet_eps=0.0,

    early_stop_enabled=True,
    early_stop_check_every=8,
    early_stop_min_visits=48,
    early_stop_best_ratio=0.7,

    dynamic_budget_enabled=True,
    dynamic_min_sims=24,
    dynamic_sharp_moves_threshold=12,
    dynamic_wide_moves_threshold=40,
    dynamic_sharp_scale=1.5,
    dynamic_wide_scale=0.5,
    dynamic_opening_scale=0.6,
    repeat_penalty_at_root=0.10
)

# --- live_log.py (fixed & VS Code friendly) --------------------
from pathlib import Path
import sys, atexit, faulthandler

# tqdm output stream (real terminal)
TQDM_OUT = getattr(sys, "__stdout__", sys.stdout)

def setup_live_log(logdir="logs", prefix="run", fsync_every=200):
    Path(logdir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    logfile = Path(logdir) / f"{prefix}_{ts}.log"
    f = open(logfile, "a", buffering=1, encoding="utf-8")

    # Enable faulthandler to REAL file handle before teeing
    try:
        if hasattr(faulthandler, "is_enabled"):
            if not faulthandler.is_enabled():
                faulthandler.enable(file=f, all_threads=True)
        else:
            faulthandler.enable(file=f, all_threads=True)
    except Exception:
        pass

    class Tee(io.TextIOBase):
        def __init__(self, *streams, fsync_every=0, file_for_fsync=None, tty=False):
            self.streams = streams
            self._lines = 0
            self._fsync_every = fsync_every
            self._file_for_fsync = file_for_fsync
            self._tty = bool(tty)

        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass
            if "\n" in data and self._fsync_every and self._file_for_fsync is not None:
                self._lines += data.count("\n")
                if self._lines >= self._fsync_every:
                    try:
                        self._file_for_fsync.flush()
                        os.fsync(self._file_for_fsync.fileno())
                    except Exception:
                        pass
                    self._lines = 0
            return len(data)

        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass
            if self._file_for_fsync is not None:
                try:
                    self._file_for_fsync.flush()
                except Exception:
                    pass

        def isatty(self):
            # Report real TTY status so tqdm animates in VS Code terminal
            return bool(getattr(sys.__stdout__, "isatty", lambda: False)())

        def fileno(self):
            if self._file_for_fsync is not None and hasattr(self._file_for_fsync, "fileno"):
                return self._file_for_fsync.fileno()
            raise io.UnsupportedOperation("fileno")

    sys.stdout = Tee(sys.stdout, f, fsync_every=fsync_every, file_for_fsync=f)
    sys.stderr = Tee(sys.stderr, f, fsync_every=fsync_every, file_for_fsync=f)
    atexit.register(lambda: (f.flush(), os.fsync(f.fileno()) if f and not f.closed else None))

    print(f"[log] Writing live log to {logfile}")
    return str(logfile)

# -------------------- Move indexing (64x64) --------------------
Move = Tuple[int, int, int, int]
# --- NEW: 64×73 factoring (from-square × 73 deltas) ---
DELTA73: List[Tuple[int,int]] = []
# 56 queen-like deltas (N,E,S,W, diagonals) with up to 7 steps
for dr, dc in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
    for k in range(1,8):
        DELTA73.append((dr*k, dc*k))
# 8 knight jumps
DELTA73 += [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]
# 9 promotions: forward, cap-left, cap-right × (N,B,R) in this order
PROMO_ORDER = [(0,1),(-1,1),(1,1)]   # (dc,dr) in board coords (rows increase down)
PROMO_PIECES = ["N","B","R"]
# NOTE: Underpromotions are not encoded separately in the 73-delta basis.
# We append the same (dr,dc) three times to reserve 9 slots, but move_to_pair()
# uses DELTA73.index((dr,dc)) so duplicates collapse to the first index.
# This is intentional for auto-queen behavior; the extra slots are reserved.
for dc,dr in PROMO_ORDER:
    for _p in PROMO_PIECES:
        DELTA73.append((dr, dc))
# Sanity: We intentionally keep 9 promotion slots; env auto-queens so duplicates in DELTA73 are OK.
assert len(DELTA73) == 73, f"Expected 73 deltas, got {len(DELTA73)}"

def sq_index(r: int, c: int) -> int:
    return r*8 + c

def move_to_pair(fr:int, fc:int, tr:int, tc:int) -> Tuple[int,int]:
    """Return (from_idx ∈ [0..63], delta_idx ∈ [0..72]); raises if not representable."""
    fr_idx = sq_index(fr, fc)
    dr, dc = tr - fr, tc - fc
    try:
        d_idx = DELTA73.index((dr, dc))
    except ValueError:
        raise ValueError(f"move not in 73-delta basis: {(fr,fc,tr,tc)}")
    return fr_idx, d_idx

def pair_to_move(fr_idx:int, d_idx:int) -> Move:
    fr, fc = divmod(fr_idx, 8)
    dr, dc = DELTA73[d_idx]
    return (fr, fc, fr+dr, fc+dc)

def index_to_move(idx: int) -> Tuple[int,int,int,int]:
    frfc, trtc = divmod(idx, 64)
    fr, fc = divmod(frfc, 8)
    tr, tc = divmod(trtc, 8)
    return (fr, fc, tr, tc)

# --------------------- State encoder (18 planes) ---------------------
def encode_state(env: ChessEnv) -> np.ndarray:
    """
    [18,8,8] float32:
      0..5 :  white P,N,B,R,Q,K
      6..11:  black P,N,B,R,Q,K
      12   :  side-to-move (1 for white else 0)
      13..16: castling rights (wK,wQ,bK,bQ) replicated
      17   :  en-passant file (1 on that file, else 0)
    """
    board = np.asarray(env.board, dtype=np.int8)         # [8,8]
    x = np.zeros((18, 8, 8), dtype=np.float32)

    # piece planes
    for ap, plane_w, plane_b in [(1,0,6),(2,1,7),(3,2,8),(4,3,9),(5,4,10),(6,5,11)]:
        x[plane_w][board ==  ap] = 1.0
        x[plane_b][board == -ap] = 1.0

    # side to move
    x[12, :, :] = 1.0 if env.current_player == 1 else 0.0

    # castling rights -> planes 13..16 (FOUR planes)
    wk = wq = bk = bq = False
    cr = getattr(env, "castling_rights", None)
    if isinstance(cr, dict):
        wk = bool(cr.get("K", False))
        wq = bool(cr.get("Q", False))
        bk = bool(cr.get("k", False))
        bq = bool(cr.get("q", False))
    x[13:17] = np.array(
        [1.0 if wk else 0.0, 1.0 if wq else 0.0, 1.0 if bk else 0.0, 1.0 if bq else 0.0],
        dtype=np.float32
    )[:, None, None]

    # en passant file -> plane 17
    ep = getattr(env, "en_passant_square", None)
    if ep:
        _, file_c = ep
        if 0 <= file_c < 8:
            x[17, :, file_c] = 1.0
    return x

# --------------------- Tactical helpers (push/pop only) ---------------------
def _exists_mate_in_one(env: ChessEnv) -> bool:
    try:
        for mv in env.legal_moves():
            env.push(mv)
            try:
                side = env.current_player
                in_check = env.in_check(side)
                has_legal = len(env.legal_moves()) > 0
                if in_check and not has_legal:
                    return True
            finally:
                env.pop()
    except Exception:
        pass
    return False

def _nonking_material_sum(env: ChessEnv) -> int:
    W = {1:1,2:3,3:3,4:5,5:9,6:0}
    s = 0
    for r in range(8):
        for c in range(8):
            p = env.board[r][c]
            if p:
                s += W[abs(p)]
    return s

def _is_endgame(env: ChessEnv) -> bool:
    s = _nonking_material_sum(env)
    no_queens = True
    for r in range(8):
        for c in range(8):
            if abs(env.board[r][c]) == 5:
                no_queens = False
                break
        if not no_queens:
            break
    return s <= 14 or no_queens

def _mobility_score(env: ChessEnv, side: int) -> int:
    bak = env.current_player
    env.current_player = side
    try:
        try:
            return len(env.legal_moves_fast())
        except Exception:
            return len(env.legal_moves())
    except Exception:
        return 0
    finally:
        env.current_player = bak

# --------------------- SEE-lite (very cheap) ---------------------
def _see_lite_after(env: ChessEnv, mv: Move) -> float:
    """
    Cheap exchange heuristic centered on destination square.
    Uses push/pop; returns approx material swing (positive = good).
    Only compute attack maps for captures or promotions (speed).
    """
    W = {1:1,2:3,3:3,4:5,5:9,6:0}
    fr, fc, tr, tc = mv
    piece = env.board[fr][fc]
    ap = abs(piece)
    target_val = W[abs(env.board[tr][tc])] if env.board[tr][tc]*piece < 0 else 0
    is_promo = (ap == 1 and ((piece > 0 and tr == 7) or (piece < 0 and tr == 0)))
    if target_val == 0 and not is_promo:
        return 0.0
    env.push(mv)
    try:
        if not hasattr(env, "attacks_of"):
            return 0.0
        mover = -env.current_player
        try:
            opp_att = env.attacks_of(env.current_player)
            my_def  = env.attacks_of(mover)
        except Exception:
            return 0.0
        attacked = (tr,tc) in opp_att
        defended = (tr,tc) in my_def
        if attacked and not defended:
            return target_val - W[ap]
        if target_val > 0 and (not attacked or defended):
            return target_val * 0.5
        return 0.0
    finally:
        env.pop()

# --------------------- Heuristic policy (push/pop) ---------------------
def heuristic_policy_push(env: ChessEnv, fast: bool = True) -> Dict[Move, float]:
    W = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    priors: Dict[Move, float] = {}
    try:
        moves = env.legal_moves_fast()
    except Exception:
        moves = env.legal_moves()
    if not moves:
        return priors

    def _nonking_material_sum_local() -> int:
        s = 0
        for r in range(8):
            for c in range(8):
                p = env.board[r][c]
                if p:
                    s += W[abs(p)]
        return s

    no_queens = True
    for r in range(8):
        for c in range(8):
            if abs(env.board[r][c]) == 5:
                no_queens = False
                break
        if not no_queens:
            break

    is_eg = (_nonking_material_sum_local() <= 14) or no_queens
    branch = len(moves)
    do_heavy = (not fast) or is_eg or (branch <= 12)

    wk = bk = None
    try:
        for r in range(8):
            for c in range(8):
                v = env.board[r][c]
                if v ==  6: wk = (r,c)
                elif v == -6: bk = (r,c)
    except Exception:
        pass

    for fr, fc, tr, tc in moves:
        piece = env.board[fr][fc]
        ap = abs(piece)
        target = env.board[tr][tc]
        s = 1e-6

        dr, dc = abs(tr-3.5), abs(tc-3.5)
        s += 0.4 * max(0.0, 1.0 - math.hypot(dr, dc)/5.0)
        if target * piece < 0:
            s += 1.0 + (W[abs(target)] - 0.15*W[ap])
        if ap == 6 and fr == tr and abs(tc-fc) == 2: s += 4.0
        elif ap == 6: s -= 3.0
        if ap == 1 and ((piece > 0 and tr == 7) or (piece < 0 and tr == 0)):
            s += 1.5

        if is_eg:
            try:
                my_side = env.current_player
                enemy_king = bk if my_side == 1 else wk
                my_king    = wk if my_side == 1 else bk
                if enemy_king:
                    edge_d = min(enemy_king[0], enemy_king[1], 7-enemy_king[0], 7-enemy_king[1])
                    s += 0.6 * (3 - edge_d)
                if enemy_king and my_king:
                    d = max(abs(my_king[0]-enemy_king[0]), abs(my_king[1]-enemy_king[1]))
                    s += 0.2 * (7 - d)
            except Exception:
                pass

        if not do_heavy:
            priors[(fr,fc,tr,tc)] = max(1e-9, s)
            continue

        env.push((fr,fc,tr,tc))
        try:
            opp = env.current_player
            try:
                opp_legal = env.legal_moves_fast()
            except Exception:
                opp_legal = env.legal_moves()
            opp_in_check = env.in_check(opp)
            if opp_in_check and len(opp_legal) == 0:
                priors[(fr,fc,tr,tc)] = 200.0
                continue
            if (not opp_in_check) and len(opp_legal) == 0:
                s -= 50.0
            if opp_in_check:
                s += 0.6

            is_promo = (ap == 1 and ((piece > 0 and tr == 7) or (piece < 0 and tr == 0)))
            if (target * piece < 0) or is_promo:
                try:
                    mover = -env.current_player
                    opp_att = env.attacks_of(env.current_player)
                    my_def  = env.attacks_of(mover)
                    if (tr,tc) in opp_att and (tr,tc) not in my_def:
                        s -= 0.9 * W[ap]
                except Exception:
                    pass

            if (is_eg or branch <= 12):
                opp_has_m1 = False
                for r2,c2,r3,c3 in opp_legal:
                    env.push((r2,c2,r3,c3))
                    try:
                        us = env.current_player
                        try:
                            us_legal = env.legal_moves_fast()
                        except Exception:
                            us_legal = env.legal_moves()
                        us_in_check = env.in_check(us)
                        if us_in_check and len(us_legal) == 0:
                            opp_has_m1 = True
                            break
                    finally:
                        env.pop()
                if opp_has_m1:
                    s -= 100.0

            if getattr(env, "halfmove_clock", 0) >= 80:
                if target*piece<0 or ap==1:
                    s += 0.4

            s += 0.4 * _see_lite_after(env, (fr,fc,tr,tc))
        finally:
            env.pop()

        priors[(fr,fc,tr,tc)] = max(1e-9, s)

    return priors

def heuristic_policy(env: ChessEnv) -> Dict[Move, float]:
    return heuristic_policy_push(env, fast=False)

# --------------------- Value heuristic ---------------------
def _material_white(env: ChessEnv) -> float:
    weights = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    score_white = 0
    for r in range(8):
        for c in range(8):
            p = env.board[r][c]
            if p:
                score_white += (weights[abs(p)] if p > 0 else -weights[abs(p)])
    return max(-1.0, min(1.0, score_white / 39.0))

def tactical_value(env: ChessEnv, perspective: int) -> float:
    side = env.current_player
    try:
        legal_now = env.legal_moves()
    except Exception:
        legal_now = []

    if not legal_now:
        if env.in_check(side):
            return 1.0 if -side == perspective else -1.0
        return 0.0

    if _exists_mate_in_one(env):
        return 1.0 if side == perspective else -1.0

    v_white = _material_white(env)

    try:
        if env.in_check(-side): v_white += 0.15 if side == 1 else -0.15
        if env.in_check( side): v_white -= 0.20 if side == 1 else -0.20
    except Exception:
        pass

    try:
        my_mob  = len(legal_now)
        opp_mob = _mobility_score(env, -side)
        mob = math.tanh((my_mob - opp_mob) / 30.0)
        v_white += 0.10 * (mob if side == 1 else -mob)
    except Exception:
        pass

    v_white = max(-1.0, min(1.0, v_white))
    return v_white if perspective == 1 else -v_white

# --------------------- Time-bank budget helper ---------------------
def compute_sim_budget_for_move(env: ChessEnv, ply: int, bank: int, nominal: int) -> Tuple[int, int]:
    """
    Returns (sims_for_this_move, new_bank).
    Bank dynamics: bank += nominal each ply; then spend 'sims'; bank = bank - sims.
    We clamp sims within [TB_MIN_SIMS_PER_MOVE, TB_MAX_SIMS_PER_MOVE] and modulate by phase/branching.
    """
    if not TB_ENABLE:
        sims = max(TB_MIN_SIMS_PER_MOVE, min(TB_MAX_SIMS_PER_MOVE, nominal))
        return sims, bank  # no bank tracking

    # Add the "salary" for this ply
    bank = int(bank + nominal)

    # Phase heuristics
    hmc = int(getattr(env, "halfmove_clock", 0))
    if ply < TB_OPENING_PLIES:
        phase_mult = TB_OPENING_MULT
    elif hmc >= TB_ENDGAME_HMC or _is_endgame(env):
        phase_mult = TB_ENDGAME_MULT
    else:
        phase_mult = TB_MIDDLEGAME_MULT

    # Branching factor
    try:
        k = len(env.legal_moves_fast())
    except Exception:
        try:
            k = len(env.legal_moves())
        except Exception:
            k = TB_BRANCH_REF
    branch_mult = min(TB_BRANCH_MAX_MULT, max(0.6, k / float(TB_BRANCH_REF)))

    # Sharpness bonus
    sharp_mult = 1.0
    try:
        if env.in_check(env.current_player):
            sharp_mult *= TB_SHARP_BONUS
    except Exception:
        pass

    # Time trouble tightening
    tt_mult = TB_TIME_TROUBLE_MULT if bank < TB_LOW_BANK_THRESHOLD else 1.0

    base = TB_NOMINAL_PER_MOVE
    sims_target = int(round(base * phase_mult * branch_mult * sharp_mult * tt_mult))
    sims_target = max(TB_MIN_SIMS_PER_MOVE, min(TB_MAX_SIMS_PER_MOVE, sims_target))

    # Don't spend more than what we actually hold (no overdraft)
    available = max(TB_MIN_SIMS_PER_MOVE, min(bank, TB_MAX_SIMS_PER_MOVE))
    sims = min(sims_target, available)

    # Spend
    bank_after = max(0, bank - sims)
    return sims, bank_after


# --------------------- Policy-only Agent ---------------------
class PolicyOnlyAgent:
    def __init__(self, policy_fn, tau: float = 1.0, eps: float = 0.0, deterministic: bool = False):
        self.policy_fn = policy_fn
        self.tau = float(tau)
        self.eps = float(eps)
        self.deterministic = bool(deterministic)
        self._last_pi: Dict[Move, float] = {}

    def train_mode(self): pass
    def eval_mode(self):  pass

    def choose_move(self, env: ChessEnv) -> Move:
        priors = self.policy_fn(env)
        moves = list(priors.keys()) if priors else env.legal_moves()
        if not moves:
            raise RuntimeError("No legal moves.")
        vals = np.array([priors.get(m, 1.0) for m in moves], dtype=np.float64)
        if self.tau > 0:
            vals = np.power(np.maximum(vals, 1e-12), 1.0 / self.tau)
        probs = vals / (vals.sum() + 1e-12)
        if self.eps > 0:
            uniform = np.full_like(probs, 1.0 / len(probs))
            probs = (1.0 - self.eps) * probs + self.eps * uniform
        probs = probs / (probs.sum() + 1e-12)
        if self.deterministic or (self.tau <= 0 and self.eps == 0.0):
            idx = int(np.argmax(probs))
        else:
            idx = int(np.random.choice(len(moves), p=probs))
        mv = moves[idx]
        self._last_pi = {m: float(p) for m, p in zip(moves, probs)}
        return mv

    def get_root_policy(self, tau: float = 1.0) -> Dict[Move, float]:
        return dict(self._last_pi)

# ------------------------ SAN / PGN helpers ------------------------
FILES = "abcdefgh"
RANKS = "12345678"
PIECE_LETTER = {1:"", 2:"N", 3:"B", 4:"R", 5:"Q", 6:"K"}

def _sq(r: int, c: int) -> str:
    return f"{FILES[c]}{r+1}"

def _is_ep(env: ChessEnv, move: Move, piece: int) -> bool:
    fr, fc, tr, tc = move
    return abs(piece) == 1 and fc != tc and env.en_passant_square == (tr, tc) and env.board[tr][tc] == 0

def san_list_from_moves(moves: List[Move]) -> List[str]:
    env = ChessEnv()
    san_moves: List[str] = []
    for mv in moves:
        fr, fc, tr, tc = mv
        piece = env.board[fr][fc]
        ap = abs(piece)
        target = env.board[tr][tc]
        dest = _sq(tr, tc)
        if ap == 6 and fr == tr and abs(tc - fc) == 2:
            san = "O-O" if tc > fc else "O-O-O"
        else:
            capture = (target * piece < 0) or _is_ep(env, mv, piece)
            is_promo = (ap == 1 and ((piece > 0 and tr == 7) or (piece < 0 and tr == 0)))
            if ap == 1:
                san = f"{FILES[fc]}x{dest}" if capture else dest
                if is_promo: san += "=Q"
            else:
                legal = env.legal_moves()
                letter = PIECE_LETTER[ap]
                others: List[Move] = []
                for (r2, c2, rto, cto) in legal:
                    if (r2, c2, rto, cto) == mv:
                        continue
                    p2 = env.board[r2][c2]
                    if p2 * env.current_player <= 0:
                        continue
                    if abs(p2) == ap and rto == tr and cto == tc:
                        others.append((r2, c2, rto, cto))
                disamb = ""
                need_file = any(c2 != fc for (r2, c2, _, _) in others)
                need_rank = any(r2 != fr for (r2, _, _, _) in others)
                if need_file:
                    disamb += FILES[fc]
                if need_rank:
                    disamb += RANKS[fr]
                san = f"{letter}{disamb}{'x' if capture else ''}{dest}"
                if is_promo: san += "=Q"
        _, _, done, info = env.step(mv)
        if done and info.get("result") == "checkmate":
            san += "#"
        elif env.in_check(env.current_player):
            san += "+"
        san_moves.append(san)
        if done:
            break
    return san_moves

def map_termination_and_result(env_after: ChessEnv, info: Dict[str,str]) -> Tuple[str, str]:
    term = info.get("result", "unknown")
    if term == "checkmate":
        winner = -env_after.current_player
        result_str = "1-0" if winner == 1 else "0-1"
        return "checkmate", result_str
    if term == "resign":
        winner = -env_after.current_player
        result_str = "1-0" if winner == 1 else "0-1"
        return "resign", result_str
    draw_terms = {"stalemate","repetition","50-move","insufficient","quickdraw","draw"}
    if term in draw_terms:
        return term, "1/2-1/2"
    return term, "1/2-1/2"

def pgn_from_game(moves: List[Move], phase: str, game_idx: int,
                  termination: str, result_str: str, san_moves = None) -> Tuple[str,int]:
    san_moves = san_moves or san_list_from_moves(moves)
    ply_count = len(san_moves)
    date_tag = datetime.date.today().strftime("%Y.%m.%d")
    tags = [
        f'[Event "Self-Play {phase} {game_idx}"]',
        f'[Site "Local"]',
        f'[Date "{date_tag}"]',
        f'[Round "{game_idx}"]',
        f'[White "Agent (W)"]',
        f'[Black "Agent (B)"]',
        f'[Result "{result_str}"]',
        f'[Termination "{termination}"]',
        f'[PlyCount "{ply_count}"]',
        f'[Variant "Standard"]',
        f'[TimeControl "-"]',
    ]
    body_parts = []
    for idx, san in enumerate(san_moves, start=1):
        if idx % 2 == 1:
            body_parts.append(f"{(idx+1)//2}. {san}")
        else:
            body_parts[-1] += f" {san}"
    body = " ".join(body_parts) + f" {result_str}"
    return "\n".join(tags) + "\n\n" + body + "\n", ply_count

# -------------------- Centralized inference server --------------------
_CENTRAL_REQ = None
_CENTRAL_RESQ = None
_CENTRAL_PROC = None
_CENTRAL_WEIGHTS = None
_CENTRAL_DEVICE = None
_WORKER_REQ = None
_WORKER_RESQ = None
_WORKER_PENDING: Dict[int, Tuple[str, object]] = {}

def _effective_fp16(device: str) -> bool:
    return (str(device).lower() != "cpu") and bool(NET_FP16)

def _central_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

from collections import OrderedDict
class _LRU:
    def __init__(self, maxn=200_000):
        self.d = OrderedDict()
        self.maxn = maxn
    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.move_to_end(k)
        return v
    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.maxn:
            self.d.popitem(last=False)

def _hash_state(state: np.ndarray) -> bytes:
    import hashlib
    h = hashlib.blake2b(state.tobytes(), digest_size=16)
    return h.digest()

def _inference_server_proc(req_q, res_q, weights_path: str, device: str, use_half: bool,
                           flush_ms: int, max_batch: int):
    import torch
    from model2 import PolicyValueNetFactored as PolicyValueNet
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    net = PolicyValueNet().to(device)
    sd = torch.load(weights_path, map_location=device)
    net.load_state_dict(sd)
    net = net.half() if use_half else net.float()
    net.eval()

    dtype = torch.float16 if use_half else torch.float32
    base_flush_s = max(0.0, float(flush_ms)) / 1000.0

    cache = _LRU(200_000)

    while True:
        batch = []
        try:
            item = req_q.get(timeout=0.1)
        except _queue.Empty:
            continue
        if item == "__STOP__":
            break
        batch.append(item)

        t0 = time.time()
        flush_s = base_flush_s
        while (time.time() - t0) < flush_s and len(batch) < max_batch:
            try:
                nxt = req_q.get_nowait()
                if nxt == "__STOP__":
                    req_q.put("__STOP__")
                    break
                batch.append(nxt)
                if len(batch) > max_batch // 2:
                    flush_s = base_flush_s * 0.5
            except _queue.Empty:
                time.sleep(0.0005)
                continue

        uniq_payloads = []
        back_refs = []
        for b in batch:
            rid = b["rid"]; kind = b["kind"]; st = b["state"]
            h = _hash_state(st)
            key = ("p", h, tuple(b["pairs"])) if kind == "policy" else ("v", h, int(b["persp"]))
            cached = cache.get(key)
            if cached is not None:
                res_q.put((rid, kind, cached))
                continue
            found = None
            for i, up in enumerate(uniq_payloads):
                if up["key"] == key:
                    found = i; break
            if found is None:
                b2 = dict(b); b2["key"] = key
                uniq_payloads.append(b2)
                back_refs.append([(rid, kind)])
            else:
                back_refs[found].append((rid, kind))

        if not uniq_payloads:
            continue

        xs = [u["state"] for u in uniq_payloads]
        xs = np.stack(xs, axis=0).astype(np.float32, copy=False)
        with torch.no_grad():
            xt = torch.as_tensor(xs, dtype=dtype, device=device)
            from_logits, delta_logits, v = net(xt)
            from_logits = from_logits.float().cpu().numpy()        # [B,64]
            delta_logits = delta_logits.float().cpu().numpy()      # [B,64,73]
            v = v.float().cpu().numpy().reshape(-1)

        for i, u in enumerate(uniq_payloads):
            if u["kind"] == "policy":
                pairs = np.asarray(u["pairs"], dtype=np.int64)     # [M,2] of (from,delta)
                if pairs.size == 0:
                    probs = []
                else:
                    frm = pairs[:,0]; dlt = pairs[:,1]
                    ll = from_logits[i][frm] + delta_logits[i][frm, dlt]   # [M]
                    probs = _central_softmax(ll[None, :])[0].astype(np.float32).tolist()
                cache.put(u["key"], probs)
                for (rid, kind) in back_refs[i]:
                    res_q.put((rid, kind, probs))
            else:
                persp = int(u["persp"])
                val = float(v[i])
                val = val if persp == 1 else -val
                cache.put(u["key"], val)
                for (rid, kind) in back_refs[i]:
                    res_q.put((rid, kind, val))

def _start_central_server(weights_path: str, device: str):
    global _CENTRAL_REQ, _CENTRAL_RESQ, _CENTRAL_PROC, _CENTRAL_WEIGHTS, _CENTRAL_DEVICE
    if not CENTRAL_INFERENCE:
        return
    if _CENTRAL_PROC is not None and _CENTRAL_PROC.is_alive():
        if _CENTRAL_WEIGHTS == weights_path and _CENTRAL_DEVICE == device:
            return
        try:
            _CENTRAL_REQ.put("__STOP__")
        except Exception:
            pass
        _CENTRAL_PROC.join(timeout=2.0)
        _CENTRAL_PROC = None

    ctx_name = "spawn" if os.name == "nt" else "fork"
    ctx = mp.get_context(ctx_name)
    _CENTRAL_REQ  = ctx.Queue(maxsize=8192)
    _CENTRAL_RESQ = ctx.Queue(maxsize=8192)

    use_half = _effective_fp16(device)
    _CENTRAL_PROC = ctx.Process(
        target=_inference_server_proc,
        args=(_CENTRAL_REQ, _CENTRAL_RESQ, weights_path, device, use_half, int(CENTRAL_FLUSH_MS), int(CENTRAL_MAX_BATCH)),
        daemon=True
    )
    _CENTRAL_PROC.start()
    _CENTRAL_WEIGHTS = weights_path
    _CENTRAL_DEVICE = device

def _stop_central_server():
    global _CENTRAL_PROC
    if _CENTRAL_PROC is not None and _CENTRAL_PROC.is_alive():
        try:
            _CENTRAL_REQ.put("__STOP__")
        except Exception:
            pass
        _CENTRAL_PROC.join(timeout=2.0)
    _CENTRAL_PROC = None

def _set_central_client(req, resq):
    global _WORKER_REQ, _WORKER_RESQ
    _WORKER_REQ, _WORKER_RESQ = req, resq

def _central_request(kind: str, state: np.ndarray, pairs: Optional[List[Tuple[int,int]]] = None, perspective: int = 1):
    rid = (os.getpid() << 32) ^ random.getrandbits(31) ^ (int(time.time()*1e6) & 0xFFFFFFFF)
    payload = {"rid": rid, "kind": kind, "state": state, "pairs": pairs or [], "persp": int(perspective)}
    _WORKER_REQ.put(payload)
    while True:
        if rid in _WORKER_PENDING:
            tag, data = _WORKER_PENDING.pop(rid)
            return tag, data
        try:
            r_rid, tag, data = _WORKER_RESQ.get(timeout=0.2)
        except _queue.Empty:
            continue
        if r_rid == rid:
            return tag, data
        else:
            _WORKER_PENDING[r_rid] = (tag, data)

def _policy_central(env: ChessEnv) -> Dict[Move, float]:
    try:
        legal = env.legal_moves()
    except Exception:
        legal = []
    if not legal:
        return {}
    pairs = [move_to_pair(fr, fc, tr, tc) for (fr,fc,tr,tc) in legal]
    # Optional safety: if env exposes factored masks, drop pairs that are masked-out
    try:
        if hasattr(env, "masks_factored"):
            from_mask, delta_mask = env.masks_factored()  # [64], [64,73] uint8/bool
            filtered = []
            kept_moves = []
            for mv, (f, d) in zip(legal, pairs):
                if from_mask[f] and delta_mask[f, d]:
                    filtered.append((f, d))
                    kept_moves.append(mv)
            if filtered:
                pairs, legal = filtered, kept_moves
            else:
                # fall back to unfiltered legal list
                pass
    except Exception:
        pass
    state = encode_state(env)
    tag, probs = _central_request("policy", state, pairs, perspective=env.current_player)
    pri = {}
    if tag == "policy":
        for m, p in zip(legal, probs):
            pri[m] = float(p)
    return pri

def _value_central(env: ChessEnv, perspective: int) -> float:
    state = encode_state(env)
    tag, val = _central_request("value", state, None, perspective=perspective)
    return float(val) if tag == "value" else 0.0

# -------------------- Backend builder (heuristic or NN) --------------------
def build_backend(use_net: bool,
                  net_weights_path: str,
                  net_device: str,
                  prefer_batched: bool = True):
    if not use_net:
        def pol(env: ChessEnv):
            return heuristic_policy_push(env, fast=True)
        def val(env: ChessEnv, perspective: int):
            return tactical_value(env, perspective)
        return pol, val

    if CENTRAL_INFERENCE and (_WORKER_REQ is not None) and (_WORKER_RESQ is not None) and prefer_batched:
        def pol(env: ChessEnv):
            return _policy_central(env)
        def val(env: ChessEnv, perspective: int):
            return _value_central(env, perspective)
        return pol, val

    import torch
    from model2 import PolicyValueNetFactored as PolicyValueNet
    _local = {"net": None, "dtype": None, "device": None}
    def _ensure():
        if _local["net"] is None:
            net = PolicyValueNet().to(net_device)
            sd = torch.load(net_weights_path, map_location=net_device)
            net.load_state_dict(sd)
            use_half = _effective_fp16(net_device)
            net = net.half() if use_half else net.float()
            net.eval()
            _local["net"] = net
            _local["dtype"] = torch.float16 if use_half else torch.float32
            _local["device"] = net_device

    def pol(env: ChessEnv):
        import torch
        _ensure()
        legal = env.legal_moves()
        if not legal:
            return {}
        x = encode_state(env)
        with torch.no_grad():
            xt = torch.as_tensor(x[None, ...], dtype=_local["dtype"], device=_local["device"])
            from_logits, delta_logits, _ = _local["net"](xt)
            from_logits  = from_logits.float().cpu().numpy()[0]      # [64]
            delta_logits = delta_logits.float().cpu().numpy()[0]     # [64,73]
        pairs = [move_to_pair(*m) for m in legal]
        # Optional safety: mask out forbidden (from, delta) if env provides masks
        try:
            if hasattr(env, "masks_factored"):
                from_mask, delta_mask = env.masks_factored()  # [64], [64,73]
                filtered = []
                kept_moves = []
                for mv, (f, d) in zip(legal, pairs):
                    if from_mask[f] and delta_mask[f, d]:
                        filtered.append((f, d))
                        kept_moves.append(mv)
                if filtered:
                    pairs, legal = filtered, kept_moves
        except Exception:
            pass
        ll = np.array([from_logits[f] + delta_logits[f, d] for (f,d) in pairs], dtype=np.float32)
        ll -= ll.max()
        probs = np.exp(ll); probs = probs / (probs.sum() + 1e-12)
        return {m: float(p) for m, p in zip(legal, probs)}

    def val(env: ChessEnv, perspective: int):
        import torch
        _ensure()
        x = encode_state(env)
        with torch.no_grad():
            xt = torch.as_tensor(x[None, ...], dtype=_local["dtype"], device=_local["device"])
            _, _, v = _local["net"](xt)
            v = float(v.float().cpu().numpy().reshape(-1)[0])
        return v if perspective == 1 else -v

    return pol, val

# -------------------- Targets (Z) with state-aware draws --------------------
def compute_z_labels(info: Dict[str, str],
                     final_env_current_player: int,
                     players: List[int],
                     final_env: ChessEnv) -> np.ndarray:
    """
    Outcome labels Z_t for each recorded state, POV=White (+1).
    - Win/loss: ±1 propagated to each POV in `players`.
    - Draws (incl. quickdraw/unknown/stalemate/3fold): base = -DRAW_PENALTY,
      plus a stronger material-shaped nudge toward the side with more material.
    """
    term = info.get("result", "draw")

    # Decisive outcomes
    if term == "checkmate" or term == "resign":
        winner = -final_env_current_player  # the side that just delivered mate / triggered resign
        outcome = 1.0 if winner == 1 else -1.0
    else:
        # treat all non-decisive terminals as draws (stalemate/threefold/insufficient/quickdraw/unknown)
        outcome = 0.0

    # Draw shaping: stronger, material-aware nudge
    draw_nudge = 0.0
    if outcome == 0.0:
        mat_white = _material_white(final_env)  # >0 means White up material
        if mat_white != 0:
            sign = 1.0 if mat_white > 0 else -1.0
            # Stronger shaping than before: saturates smoothly with material gap
            draw_nudge = 0.20 * float(np.tanh(abs(mat_white) / 4.0)) * sign

    Z = np.empty((len(players),), dtype=np.float32)
    for i, p in enumerate(players):
        if outcome == 0.0:
            base = -DRAW_PENALTY
            # POV adjustment: White POV sees +nudge when White is better; Black POV sees -nudge
            adj = draw_nudge if p == 1 else -draw_nudge
            Z[i] = base + adj
        else:
            # Propagate decisive result to each POV
            Z[i] = outcome if p == 1 else -outcome

    return Z

# -------------------- JSONL helper --------------------
def _append_jsonl(path: str, obj: dict):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

# -------------------- Atomic save helpers (Windows-safe) --------------------
def _atomic_savez(path: str, **arrays):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)   # pass file handle to avoid '.npz' auto-append
        try:
            f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)

def _atomic_savez_fast(path: str, **arrays):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)              # pass file handle to avoid '.npz' auto-append
        try:
            f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)

# ---------------------- Opening book (tiny) ----------------------
def _uci_to_move(uci: str) -> Move:
    # uci like "e2e4" or "g1f3"; we ignore promotions (auto-queen already in env)
    f_file, f_rank, t_file, t_rank = uci[0], uci[1], uci[2], uci[3]
    fc = FILES.index(f_file); fr = int(f_rank) - 1
    tc = FILES.index(t_file); tr = int(t_rank) - 1
    # env uses rows 0..7 with white pawns at row=1 (rank 2) and WHITE moves "down" (+r)
    return (fr, fc, tr, tc)

# A few short skeletons (8 plies max). Feel free to extend.
_OPENING_BOOK_UCI: List[List[str]] = [
    # ----- Open Games -----
    # Italian (Giuoco Piano)
    ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","c2c3","g8f6"],
    ["e2e4","e7e5","g1f3","b8c6","f1c4","g8f6","d2d3","f8c5"],
    # Two Knights (quiet)
    ["e2e4","e7e5","g1f3","b8c6","f1c4","g8f6","d2d3","f8c5"],
    # Two Knights (main tactical branch start)
    ["e2e4","e7e5","g1f3","b8c6","f1c4","g8f6","f3g5","d7d5"],

    # Ruy Lopez
    ["e2e4","e7e5","g1f3","b8c6","f1b5","a7a6","b5a4","g8f6"],
    ["e2e4","e7e5","g1f3","b8c6","f1b5","g7g6","c2c3","f8g7"],

    # Scotch
    ["e2e4","e7e5","g1f3","b8c6","d2d4","e5d4","f3d4","g8f6"],

    # ----- Semi-open (Sicilian, Caro-Kann, Scandinavian) -----
    # Sicilian Najdorf shell
    ["e2e4","c7c5","g1f3","d7d6","d2d4","c5d4","f3d4","g8f6"],
    # Sicilian Taimanov shell
    ["e2e4","c7c5","g1f3","e7e6","d2d4","c5d4","f3d4","b8c6"],
    # Sicilian Kan/Paulsen shell
    ["e2e4","c7c5","g1f3","e7e6","d2d4","c5d4","f3d4","a7a6"],

    # Caro-Kann Advance
    ["e2e4","c7c6","d2d4","d7d5","e4e5","f7f5","g2g3","e7e6"],
    ["e2e4","c7c6","d2d4","d7d5","e4e5","c8f5","g1f3","e7e6"],
    # Caro-Kann Classical (4...Bf5)
    ["e2e4","c7c6","d2d4","d7d5","b1c3","d5e4","c3e4","c8f5"],
    ["e2e4","c7c6","d2d4","d7d5","b1c3","d5e4","c3e4","g8f6"],
    # Caro-Kann Exchange
    ["e2e4","c7c6","d2d4","d7d5","e4d5","c6d5","g1f3","b8c6"],

    # Scandinavian
    ["e2e4","d7d5","e4d5","d8d5","b1c3","d5a5","d2d4","c7c6"],

    # ----- French & Pirc/KID shells for variety -----
    # French Tarrasch shell
    ["e2e4","e7e6","d2d4","d7d5","b1d2","c7c5","g1f3","b8c6"],
    # Pirc
    ["e2e4","d7d6","d2d4","g8f6","b1c3","g7g6","g1f3","f8g7"],

    # ----- Closed (d4) -----
    # QGD
    ["d2d4","d7d5","c2c4","e7e6","g1f3","g8f6","b1c3","f8e7"],
    # Nimzo-Indian shell
    ["d2d4","d7d5","c2c4","e7e6","g1f3","g8f6","b1c3","f8b4"],
    # KID shell
    ["d2d4","g8f6","c2c4","g7g6","b1c3","f8g7","e2e4","d7d6"],

    # ----- English / Flank -----
    ["c2c4","e7e5","b1c3","g8f6","g1f3","b8c6","g2g3","f8b4"],
    ["c2c4","c7c5","g1f3","g8f6","g2g3","b8c6","f1g2","e7e6"],
]

_OPENING_BOOK: List[List[Move]] = [[_uci_to_move(u) for u in line] for line in _OPENING_BOOK_UCI]

def _book_next_move(env: "ChessEnv", ply: int, line: List[Move]) -> Optional[Move]:
    """If current position matches the first `ply` moves of `line`, return line[ply]."""
    try:
        hist = getattr(env, "_history", None)
        if hist is None:
            return None
        # Compare already-played moves with the book prefix
        if ply > len(line):
            return None
        for k in range(min(ply, len(line))):
            if k >= len(hist): break
            if tuple(hist[k]["move"]) != tuple(line[k]):
                return None
        # If we are exactly at ply < len(line), suggest the next
        if ply < len(line):
            mv = line[ply]
            # Must be legal in case book line clashes with env rules after deviations
            leg = env.legal_moves()
            if mv in leg:
                return mv
    except Exception:
        pass
    return None

# -------------------- Self-play (data collection) --------------------
def play_game_collect_data(agent_mcts: MCTSAgent,
                           agent_policy: PolicyOnlyAgent,
                           mode: str,
                           max_plies: int,
                           train_tau: float = 1.0,
                           open_policy_plies: int = 0,
                           game_idx: Optional[int] = None,
                           enable_ply_tqdm: bool = False,
                           quickdraw_halfmoves: Optional[int] = None,
                           anneal_tau: bool = False,
                           resign_enable: bool = True,
                           value_fn=None):
    env = ChessEnv()
    # Skip recording very-late "shuffle" positions (no captures/pawns for long)
    SKIP_LATE_SHUFFLE_THRESHOLD = 70

    # Time-bank state (per game)
    tb_bank = TB_TOTAL_BANK

    states: List[np.ndarray] = []
    PF_list: List[np.ndarray] = []
    PD_list: List[np.ndarray] = []
    players: List[int] = []
    moves: List[Move] = []
    last_info: Dict[str,str] = {}
    san_moves: List[str] = []

    desc = f"Game {game_idx}" if game_idx is not None else "Game"

    consec_bad = 0

    with tqdm(total=max_plies, desc=desc, position=1, leave=False,
              disable=not enable_ply_tqdm, mininterval=0.5, file=TQDM_OUT) as ply_bar:
        for ply in range(max_plies):
            if getattr(env, "done", False):
                break

            # --- Opening book hook (EXPLORE & POLICY modes only) ---
            mv = None
            pi_dict = None
            if (mode != "eval") and USE_OPENING_BOOK and (BOOK_PLIES > 0) and (ply < BOOK_PLIES) and (np.random.rand() < BOOK_PROB):
                line = _OPENING_BOOK[np.random.randint(len(_OPENING_BOOK))]
                mv_book = _book_next_move(env, ply, line)
                if mv_book is not None:
                    mv = mv_book
                    pi_dict = {mv_book: 1.0}  # one-hot; generic writer below will handle it

            # --- Normal selection if no book move was taken ---
            if mv is None or pi_dict is None:
                use_policy_now = (mode == "policy") or (mode == "mcts" and ply < open_policy_plies)
                if use_policy_now:
                    if anneal_tau:
                        agent_policy.tau = float(tau_schedule_explore(ply))
                        agent_policy.eps = float(EXPLORE_POLICY_EPS)
                    mv = agent_policy.choose_move(env)
                    pi_dict = agent_policy.get_root_policy(tau=train_tau)
                else:
                    # --- time-bank: per-move simulation override ---
                    orig_sims = agent_mcts.cfg.n_simulations
                    sims_for_move = orig_sims
                    if TB_ENABLE:
                        sims_for_move, tb_bank = compute_sim_budget_for_move(env, ply, tb_bank, TB_NOMINAL_PER_MOVE)
                        agent_mcts.cfg.n_simulations = sims_for_move
                    mv = agent_mcts.choose_move(env, explore=True)
                    # restore cfg to avoid leaking override elsewhere
                    agent_mcts.cfg.n_simulations = orig_sims
                    pi_dict = agent_mcts.get_root_policy(tau=train_tau)

            # --- dataset write (state + π) once (book or normal), but skip "dead" phases ---
            record_this = getattr(env, "halfmove_clock", 0) < SKIP_LATE_SHUFFLE_THRESHOLD
            if record_this:
                states.append(encode_state(env))
                players.append(int(env.current_player))

                legal = list(pi_dict.items()) if pi_dict else []
                if not legal:
                    legal = [(m, 1.0) for m in env.legal_moves()]
                probs = np.array([p for (_, p) in legal], dtype=np.float32)
                probs = probs / (probs.sum() + 1e-12)
                # --- CHANGED: dynamic label smoothing by branching factor ---
                if POLICY_SMOOTH_EPS > 0 and len(probs) > 0:
                    k = float(len(probs))
                    eps_local = float(POLICY_SMOOTH_EPS * min(1.0, (k / 40.0)))
                    if eps_local > 0.0:
                        u = np.full_like(probs, 1.0 / len(probs))
                        probs = (1.0 - eps_local) * probs + eps_local * u
                        probs = probs / (probs.sum() + 1e-12)
                # --- NEW: build factored PF/PD ---
                PF = np.zeros((64,), dtype=np.float32)
                PD = np.zeros((64, 73), dtype=np.float32)
                for (m, p) in zip((m for m,_ in legal), probs):
                    f, d = move_to_pair(*m)
                    PF[f] += float(p)
                    PD[f, d] += float(p)
                # Normalize per-from row of PD to PF (optional but nice): if PF[f]>0 keep as joint, training will read as joint.
                # Append to running lists (create them once above loop): PF_list, PD_list
                PF_list.append(PF)
                PD_list.append(PD)
            
            try:
                san = getattr(env, "san", None)
                if callable(san):
                    san_moves.append(san(mv))
            except Exception:
                pass

            moves.append(mv)
            _, _, done, info = env.step(mv)
            last_info = info

            if resign_enable and (ply+1) >= RESIGN_AFTER_PLIES and value_fn is not None:
                v_eval_pov = value_fn(env, perspective=env.current_player)
                if v_eval_pov < RESIGN_VALUE:
                    consec_bad += 1
                else:
                    consec_bad = 0
                if consec_bad >= RESIGN_CONSEC_PLY:
                    last_info = {"result": "resign"}
                    break

            if (not done) and (quickdraw_halfmoves is not None):
                if getattr(env, "halfmove_clock", 0) >= int(quickdraw_halfmoves):
                    last_info = {"result": "quickdraw"}
                    break

            if enable_ply_tqdm:
                ply_bar.update(1)

            if done:
                break

    return moves, last_info, states, (PF_list, PD_list), players

# -------------------- One game wrapper --------------------
def run_one_game(game_idx: int,
                 base_seed: int,
                 cfg: SearchConfig,
                 phase_name: str,            # "EXPLORE" | "EVAL"
                 mode: str,                  # "mcts" | "policy"
                 max_plies: int,
                 data_dir: str,
                 save_dataset: bool,
                 use_net: bool,
                 net_weights_path: str,
                 net_device: str,
                 explore_policy_tau: float = 1.0,
                 explore_policy_eps: float = 0.10,
                 open_policy_plies: int = 0,
                 quickdraw_halfmoves: Optional[int] = None):
    seed_offset = 0 if phase_name == "EXPLORE" else 10_000
    seed = base_seed + seed_offset + game_idx
    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:
        pass

    policy_fn, value_fn = build_backend(use_net, net_weights_path, net_device)

    agent_mcts = MCTSAgent(policy_fn=policy_fn, value_fn=value_fn, cfg=cfg)
    agent_policy = PolicyOnlyAgent(policy_fn=policy_fn, tau=explore_policy_tau, eps=explore_policy_eps)

    if phase_name == "EXPLORE":
        agent_mcts.train_mode()
    else:
        agent_mcts.eval_mode()

    mode_for_game = "mcts" if (phase_name == "EVAL") else mode
    moves, info, states, (PF_list, PD_list), players = play_game_collect_data(
        agent_mcts, agent_policy, mode_for_game, max_plies, train_tau=1.0,
        open_policy_plies=open_policy_plies, game_idx=game_idx,
        enable_ply_tqdm=False,
        quickdraw_halfmoves=quickdraw_halfmoves,
        anneal_tau=(phase_name == "EXPLORE"),
        resign_enable = (phase_name != "EXPLORE") or (CURRENT_CYCLE > 6),
        value_fn=value_fn
    )

    tmp_env = ChessEnv()
    for mv in moves:
        _, _, done, _ = tmp_env.step(mv)
        if done:
            break
    termination, result_str = map_termination_and_result(tmp_env, info)
    pgn, _plycount = pgn_from_game(moves, phase=phase_name, game_idx=game_idx,
                               termination=termination, result_str=result_str)

    pgn_dir = EXPLORE_DIR if phase_name == "EXPLORE" else EVAL_DIR
    pgn_path = os.path.join(pgn_dir, f"{phase_name.lower()}_game_{game_idx}.pgn")
    if WRITE_PER_GAME_PGNS:
        with open(pgn_path, "w", encoding="utf-8") as f:
            f.write(pgn)

    z = compute_z_labels(info, final_env_current_player=tmp_env.current_player, players=players, final_env=tmp_env)

    data_path = None
    if save_dataset and states:
        X = np.stack(states, axis=0)
        Z = z
                # --- Subsample to MAX_SAMPLES_PER_GAME with priority for informative plies ---
        if MAX_SAMPLES_PER_GAME and len(X) > MAX_SAMPLES_PER_GAME:
            n = len(X)
            env_pick = ChessEnv()
            interesting: List[int] = []
            tail_keep = min(16, n)
            opening_keep = min(8, n)
            for i, mv in enumerate(moves[:n]):
                fr, fc, tr, tc = mv
                piece  = env_pick.board[fr][fc] if 0 <= fr < 8 and 0 <= fc < 8 else 0
                target = env_pick.board[tr][tc] if 0 <= tr < 8 and 0 <= tc < 8 else 0
                ap = abs(piece)
                is_promo   = (ap == 1 and ((piece > 0 and tr == 7) or (piece < 0 and tr == 0)))
                is_capture = (target * piece < 0) or _is_ep(env_pick, mv, piece)
                hmc = int(getattr(env_pick, "halfmove_clock", 0))
                env_pick.push(mv)
                gives_check = False
                try:
                    gives_check = bool(env_pick.in_check(env_pick.current_player))
                except Exception:
                    pass
                env_pick.pop()
                if is_capture or is_promo or gives_check or hmc < 40:
                    interesting.append(i)
            seen = set()
            interesting = [i for i in interesting if not (i in seen or seen.add(i))]
            prefix = list(range(opening_keep))
            tail   = list(range(max(0, n - tail_keep), n))
            base = []
            for i in prefix + interesting + tail:
                if i not in seen:
                    seen.add(i); base.append(i)
            base = sorted(set(base))
            need = MAX_SAMPLES_PER_GAME - len(base)
            if need > 0:
                mask = np.ones(n, dtype=bool)
                mask[np.array(base, dtype=int)] = False
                pool = np.flatnonzero(mask)
                if pool.size > 0:
                    extra = np.linspace(0, pool.size - 1, min(need, pool.size)).astype(np.int64)
                    base.extend(pool[extra].tolist())
            sel = np.array(sorted(base[:MAX_SAMPLES_PER_GAME]), dtype=np.int64)

            # Slice X/Z and the factored labels in lockstep
            X  = X[sel]
            Z  = Z[sel]
            PF_list = [PF_list[i] for i in sel.tolist()]
            PD_list = [PD_list[i] for i in sel.tolist()]
        data_path = os.path.join(data_dir, f"data_{phase_name.lower()}_game_{game_idx}.npz")
        PF = np.stack(PF_list, axis=0).astype(np.float32)
        PD = np.stack(PD_list, axis=0).astype(np.float32)
        _atomic_savez_fast(data_path, X=X, Z=Z, PF=PF, PD=PD)

    _append_jsonl(GAMES_JSONL, {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "phase": phase_name,
        "game": game_idx,
        "termination": termination,
        "result": result_str,
        "moves": len(moves),
        "save": data_path
    })

    return game_idx, pgn, (pgn_path if WRITE_PER_GAME_PGNS else None), data_path

# -------------------- Dataset concatenation (sparse-aware) --------------------
def concat_npz_sparse(paths: List[str], out_path: str) -> bool:
    Xs, Zs, PFs, PDs = [], [], [], []
    for p in paths:
        if not p:
            continue
        with np.load(p) as d:
            Xs.append(d["X"]); Zs.append(d["Z"])
            PFs.append(d["PF"]); PDs.append(d["PD"])
    if not Xs:
        return False
    X = np.concatenate(Xs, axis=0)
    Z = np.concatenate(Zs, axis=0)
    PF = np.concatenate(PFs, axis=0)
    PD = np.concatenate(PDs, axis=0)
    _atomic_savez(out_path, X=X, Z=Z, PF=PF, PD=PD)
    return True

# --- NEW: Data augmentation helpers (8 symmetries + color flip) ---
def _sym_map_rc(sym: int, r: int, c: int) -> Tuple[int,int]:
    # 0:I, 1:R90, 2:R180, 3:R270, 4:FH, 5:FV, 6:FD (main diag), 7:FAD (anti-diag)
    if sym == 0:   return r, c
    if sym == 1:   return c, 7 - r
    if sym == 2:   return 7 - r, 7 - c
    if sym == 3:   return 7 - c, r
    if sym == 4:   return r, 7 - c
    if sym == 5:   return 7 - r, c
    if sym == 6:   return c, r
    if sym == 7:   return 7 - c, 7 - r
    return r, c

def _apply_symmetry_state(x: np.ndarray, sym: int) -> np.ndarray:
    # x: [18,8,8]; use numpy transforms on spatial dims for all planes
    def _tx(p):
        if   sym == 0: return p
        elif sym == 1: return np.rot90(p, k=1, axes=(1,0))
        elif sym == 2: return np.rot90(p, k=2, axes=(1,0))
        elif sym == 3: return np.rot90(p, k=3, axes=(1,0))
        elif sym == 4: return np.flip(p, axis=2)        # horiz
        elif sym == 5: return np.flip(p, axis=1)        # vert
        elif sym == 6: return np.swapaxes(p, 1, 2)      # main diag
        elif sym == 7: return np.rot90(np.swapaxes(p,1,2), k=2, axes=(1,2))
    y = np.stack([_tx(x[i:i+1])[0] for i in range(x.shape[0])], axis=0)
    return y

def _apply_color_flip_state(x: np.ndarray) -> np.ndarray:
    # rotate 180°, swap white/black piece planes, flip side-to-move and castling planes
    y = np.rot90(x, k=2, axes=(1,2))  # spatial 180°
    y0 = y.copy()
    # swap 0..5 (white pieces) with 6..11 (black pieces)
    y0[0:6], y0[6:12] = y[6:12], y[0:6]
    # side to move (plane 12): invert 0/1
    y0[12] = 1.0 - y[12]
    # castling planes 13..16: swap (wK,wQ,bK,bQ) -> (bK,bQ,wK,wQ)
    y0[13:17] = y[[15,16,13,14]]
    # en-passant (17) already rotated by spatial 180°
    return y0

# -------------------- Training on (X, sparse PI, Z) --------------------
def _train_on_npz(npz_path: str,
                  out_weights_path: str,
                  epochs: int = 4,
                  batch_size: int = 256,
                  lr: float = 1e-3,
                  weight_decay: float = 1e-4,
                  value_loss_weight: float = 1.0,
                  device: str = "cpu") -> bool:
    if not os.path.exists(npz_path):
        print(f"[train] dataset not found: {npz_path}")
        return False

    import torch
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from model import PolicyValueNetFactored as PolicyValueNet

    torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    class NpzSparseDataset(Dataset):
        def __init__(self, path: str):
            d = np.load(path)
            self.X = d["X"]
            self.Z = d["Z"]
            if "PF" in d and "PD" in d:
                self.PF = d["PF"].astype(np.float32)        # [N,64]
                self.PD = d["PD"].astype(np.float32)        # [N,64,73]
                self.factored = True
            else:
                raise RuntimeError("Expected PF/PD in dataset (factored policy).")
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i):
            return (self.X[i], self.Z[i], self.PF[i], self.PD[i])

    ds = NpzSparseDataset(npz_path)
    if len(ds) == 0:
        print("[train] empty dataset; skipping.")
        return False

    pin = (device != "cpu")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,        # Windows-safe
        pin_memory=pin
    )

    net = PolicyValueNet().to(device)

    # >>> NEW: initialize from previous cycle, if present
    if os.path.exists(out_weights_path):
        try:
            sd_prev = torch.load(out_weights_path, map_location=device)
            net.load_state_dict(sd_prev, strict=False)
            print("[train] initialized from previous weights")
        except Exception as e:
            print(f"[train] could not init from previous weights: {e}")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * math.ceil(len(ds) / batch_size)
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    def _lr_schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_schedule)

    # ------- EMA (float-only smoothing; ints copied)  -------
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    def _ema_update():
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if torch.is_floating_point(v):
                    ema[k].mul_(EMA_DECAY).add_(v.detach(), alpha=1.0 - EMA_DECAY)
                else:
                    # Keep integer/bool buffers in sync (e.g., num_batches_tracked)
                    ema[k].copy_(v)

    net.train()
    step = 0
    for epoch in range(1, epochs + 1):
        total, pol_loss_sum, val_loss_sum = 0, 0.0, 0.0
        t0 = time.time()
        for batch in dl:
            X = torch.as_tensor(batch[0], dtype=torch.float32, device=device)
            Z = torch.as_tensor(batch[1], dtype=torch.float32, device=device).view(-1,1)
            PF = torch.as_tensor(batch[2], dtype=torch.float32, device=device)        # [B,64]
            PD = torch.as_tensor(batch[3], dtype=torch.float32, device=device)        # [B,64,73]

            from_logits, delta_logits, v = net(X)
            logp_from  = torch.log_softmax(from_logits, dim=1)    # [B,64]
            logp_delta = torch.log_softmax(delta_logits, dim=2)   # [B,64,73]

            # Per-sample policy losses
            loss_from_i  = -(PF * logp_from).sum(dim=1)          # [B]
            loss_delta_i = -(PD * logp_delta).sum(dim=(1,2))     # [B]
            pol_loss_i   = loss_from_i + loss_delta_i            # [B]

            if PRIORITIZED_LOSS:
                # Weight by policy target entropy (higher entropy → slightly higher weight)
                with torch.no_grad():
                    # Entropy of PF + row-wise entropy of PD weighted by PF (matches factored target)
                    eps = 1e-12
                    H_from = -(PF.clamp_min(eps) * PF.clamp_min(eps).log()).sum(dim=1)  # [B]
                    # Conditional entropy: sum_f PF[f] * H(PD[f])
                    PD_safe = PD.clamp_min(eps)
                    H_rows = -(PD_safe * PD_safe.log()).sum(dim=2)                      # [B,64]
                    H_cond = (PF * H_rows).sum(dim=1)                                   # [B]
                    H = H_from + H_cond                                                 # [B]
                    # Normalize weights to mean 1.0
                    w = H / (H.mean().clamp_min(1e-6))
                policy_loss = (w * pol_loss_i).mean()
            else:
                policy_loss = pol_loss_i.mean()

            # Value loss, optionally weighted by the same w
            se_i = (v - Z).pow(2).view(-1)                       # [B]
            if PRIORITIZED_LOSS:
                value_loss = (w * se_i).mean()
            else:
                value_loss = se_i.mean()
            val_loss = value_loss

            loss = policy_loss + value_loss_weight * val_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
            opt.step()
            scheduler.step()
            _ema_update()

            bs = X.size(0)
            total += bs
            pol_loss_sum += float(policy_loss.item()) * bs
            val_loss_sum += float(value_loss.item()) * bs
            step += 1

        elapsed = time.time() - t0
        print(f"[train] epoch {epoch}/{epochs} | "
              f"policy_loss={pol_loss_sum/total:.4f} | value_loss={val_loss_sum/total:.4f} | "
              f"elapsed={elapsed:.1f}s")

    os.makedirs(os.path.dirname(out_weights_path), exist_ok=True)
    # Save EMA tensors for reproducibility / inspection
    import torch as _torch
    _torch.save(ema, out_weights_path + ".ema")
    # Load EMA into the net and save final weights
    net.load_state_dict(ema, strict=False)
    _torch.save(net.state_dict(), out_weights_path)
    print(f"[train] saved weights (EMA) -> {out_weights_path}")
    return True

# -------------------- WLD → score / elo helpers --------------------
def _score_from_wld(wins: int, draws: int, losses: int) -> float:
    n = wins + draws + losses
    if n <= 0:
        return 0.0
    return (wins + 0.5 * draws) / n

def _elo_from_score(score: float, eps: float = 1e-6) -> float:
    p = min(max(score, eps), 1.0 - eps)
    return 400.0 * math.log10(p / (1.0 - p))

def _elo_ci(score: float, n_games: int) -> Tuple[float, float]:
    if n_games <= 0:
        return 0.0, 0.0
    p = min(max(score, 1e-6), 1 - 1e-6)
    se = math.sqrt(p * (1 - p) / n_games)
    dEdp = 400.0 / math.log(10) / (p * (1 - p))
    elo_se = dEdp * se
    delta = 1.96 * elo_se
    return delta, elo_se

# -------------------- SPRT (sequential test) --------------------
def sprt_decision(wins: int, draws: int, losses: int,
                  p0=SPRT_P0, p1=SPRT_P1, alpha=SPRT_ALPHA, beta=SPRT_BETA) -> str:
    n = wins + draws + losses
    if n == 0:
        return "continue"
    s = wins + 0.5*draws
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))
    llr = s * math.log((p1 + 1e-9) / (p0 + 1e-9)) + (n - s) * math.log((1 - p1 + 1e-9) / (1 - p0 + 1e-9))
    if llr >= A:
        return "accept"
    if llr <= B:
        return "reject"
    return "continue"

# -------------------- Arena: eval vs explore --------------------
def play_arena_game(game_idx: int,
                    cfg_eval: SearchConfig,
                    cfg_explore: SearchConfig,
                    eval_backend: Tuple,
                    explore_backend: Tuple,
                    max_plies: int,
                    eval_is_white: bool,
                    open_policy_plies: int = 0,
                    quickdraw_halfmoves: int = 10_000) -> Tuple[str, str, int, str]:
    env = ChessEnv()
    (pol_eval, val_eval) = eval_backend
    (pol_ex,  val_ex)    = explore_backend
    # Time-bank state (single bank for the full game; both sides draw from it)
    tb_bank = TB_TOTAL_BANK
    agent_eval = MCTSAgent(policy_fn=pol_eval, value_fn=val_eval, cfg=cfg_eval)
    agent_ex   = MCTSAgent(policy_fn=pol_ex,  value_fn=val_ex,  cfg=cfg_explore)
    agent_eval.eval_mode()
    agent_ex.eval_mode()

    polonly_eval = PolicyOnlyAgent(pol_eval, tau=0.0, eps=0.0, deterministic=True)
    polonly_ex   = PolicyOnlyAgent(pol_ex,  tau=0.0, eps=0.0, deterministic=True)

    moves: List[Move] = []
    last_info: Dict[str, str] = {}
    ply = 0
    consec_bad_eval = 0
    consec_bad_ex   = 0

    for _ in range(max_plies):
        if getattr(env, "done", False):
            break

        side_is_eval = (env.current_player == 1 and eval_is_white) or (env.current_player == -1 and not eval_is_white)

        if ply < open_policy_plies:
            mv = (polonly_eval if side_is_eval else polonly_ex).choose_move(env)
        else:
            # --- time-bank per-move override for MCTS ---
            agent = (agent_eval if side_is_eval else agent_ex)
            orig_sims = agent.cfg.n_simulations
            sims_for_move = orig_sims
            if TB_ENABLE:
                sims_for_move, tb_bank = compute_sim_budget_for_move(env, ply, tb_bank, TB_NOMINAL_PER_MOVE)
                agent.cfg.n_simulations = sims_for_move
            mv = agent.choose_move(env, explore=False)
            agent.cfg.n_simulations = orig_sims

        moves.append(mv)
        _, _, done, info = env.step(mv)
        last_info = info
        ply += 1

        if (ply >= RESIGN_AFTER_PLIES):
            v = (val_eval if env.current_player == (1 if eval_is_white else -1) else val_ex)(env, perspective=env.current_player)
            eval_to_move = (env.current_player == (1 if eval_is_white else -1))
            if eval_to_move:
                consec_bad_eval = consec_bad_eval + 1 if v < RESIGN_VALUE else 0
            else:
                consec_bad_ex   = consec_bad_ex   + 1 if v < RESIGN_VALUE else 0
            if consec_bad_eval >= RESIGN_CONSEC_PLY or consec_bad_ex >= RESIGN_CONSEC_PLY:
                last_info = {"result": "resign"}
                break

        if not done and getattr(env, "halfmove_clock", 0) >= quickdraw_halfmoves:
            last_info = {"result": "quickdraw"}
            break

        if done:
            break

    termination, result_str = map_termination_and_result(env, last_info)
    pgn, plycount = pgn_from_game(moves, phase="ARENA", game_idx=game_idx,
                                  termination=termination, result_str=result_str)

    if termination in ("checkmate","resign"):
        winner = -env.current_player
        eval_won = ((winner == 1 and eval_is_white) or (winner == -1 and not eval_is_white))
        result_eval_pov = "win" if eval_won else "loss"
    else:
        if result_str == "1-0":
            result_eval_pov = "win" if eval_is_white else "loss"
        elif result_str == "0-1":
            result_eval_pov = "loss" if eval_is_white else "win"
        else:
            result_eval_pov = "draw"

    return pgn, result_eval_pov, plycount, termination

# -------------------- Arena chunk worker (parallel-friendly) --------------------
def _arena_run_chunk(start_g: int, end_g: int,
                     eval_use_net: bool, explore_use_net: bool,
                     eval_net_path: str, explore_net_path: str, net_device: str,
                     out_dir: str,
                     cfg_eval_runtime: SearchConfig,
                     cfg_explore_runtime: SearchConfig,
                     req=None, resq=None) -> Tuple[int, int, int, List[Tuple]]:
    if CENTRAL_INFERENCE and (req is not None) and (resq is not None):
        _set_central_client(req, resq)

    # Use local (non-central) inference so each side can load different weights
    pol_eval, val_eval = build_backend(eval_use_net,    eval_net_path,    net_device, prefer_batched=False)
    pol_ex,   val_ex   = build_backend(explore_use_net, explore_net_path, net_device, prefer_batched=False)

    eval_w = ex_w = dr = 0
    rows: List[Tuple] = []

    for g in range(start_g, end_g + 1):
        eval_is_white = (g % 2 == 1)
        pgn, result, plycount, term = play_arena_game(
            game_idx=g,
            cfg_eval=cfg_eval_runtime,
            cfg_explore=cfg_explore_runtime,
            eval_backend=(pol_eval, val_eval),
            explore_backend=(pol_ex,  val_ex),
            max_plies=ARENA_MAX_PLIES,
            eval_is_white=eval_is_white,
            open_policy_plies=OPEN_POLICY_PLIES_ARENA,
            quickdraw_halfmoves=ARENA_QUICKDRAW_HALFMOVES
        )
        with open(os.path.join(out_dir, f"arena_game_{g}.pgn"), "w", encoding="utf-8") as f:
            f.write(pgn)

        if   result == "win":  eval_w += 1
        elif result == "loss": ex_w   += 1
        else:                  dr     += 1

        rows.append((g, "white" if eval_is_white else "black", plycount, term, result))

    return eval_w, ex_w, dr, rows

# -------------------- Arena harness (SPRT + parallel-aware) --------------------
def run_arena_sprt(max_games: int,
                   eval_use_net: bool,
                   explore_use_net: bool,
                   eval_net_path: str,           # <-- CHANGED
                   explore_net_path: str,        # <-- CHANGED
                   net_device: str,
                   out_dir: str,
                   cfg_eval_runtime: SearchConfig,
                   cfg_explore_runtime: SearchConfig ) -> Tuple[int,int,int,List[Tuple]]:

    os.makedirs(out_dir, exist_ok=True)

    # (Removed central _set_central_client here; we need per-side nets)
    # Build local backends (used by the sequential path below).
    pol_eval, val_eval = build_backend(eval_use_net,    eval_net_path,    net_device, prefer_batched=False)  # <-- CHANGED
    pol_ex,   val_ex   = build_backend(explore_use_net, explore_net_path, net_device, prefer_batched=False)  # <-- CHANGED

    # ---------- Path A: SPRT enabled → sequential (original behavior) ----------
    if SPRT_USE:
        eval_w = ex_w = dr = 0
        rows_all: List[Tuple] = []
        for g in tqdm(range(1, max_games + 1),
                      desc=f"ARENA (SPRT up to {max_games})",
                      dynamic_ncols=True, leave=True, mininterval=0.2, file=TQDM_OUT):
            eval_is_white = (g % 2 == 1)
            pgn, result, plycount, term = play_arena_game(
                game_idx=g,
                cfg_eval=cfg_eval_runtime,
                cfg_explore=cfg_explore_runtime,
                eval_backend=(pol_eval, val_eval),
                explore_backend=(pol_ex, val_ex),
                max_plies=ARENA_MAX_PLIES,
                eval_is_white=eval_is_white,
                open_policy_plies=OPEN_POLICY_PLIES_ARENA,
                quickdraw_halfmoves=ARENA_QUICKDRAW_HALFMOVES
            )
            with open(os.path.join(out_dir, f"arena_game_{g}.pgn"), "w", encoding="utf-8") as f:
                f.write(pgn)

            if result == "win":
                eval_w += 1
            elif result == "loss":
                ex_w += 1
            else:
                dr += 1

            rows_all.append((g, "white" if eval_is_white else "black", plycount, term, result))

            if g >= 10:
                decision = sprt_decision(eval_w, dr, ex_w)
                if decision in ("accept", "reject"):
                    print(f"[arena][SPRT] decision={decision} at game {g} | W={eval_w}, D={dr}, L={ex_w}")
                    break

        return eval_w, ex_w, dr, rows_all

    # ---------- Path B: SPRT disabled → parallel chunked (faster, same quality) ----------
    eval_w = ex_w = dr = 0
    rows_all: List[Tuple] = []

    if ARENA_PARALLEL:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx_name = "spawn" if os.name == "nt" else "fork"
        ctx = mp.get_context(ctx_name)

        workers = max(1, ARENA_MAX_WORKERS)
        chunk = max(1, (max_games + workers - 1) // workers)
        ranges = [(s, min(max_games, s + chunk - 1)) for s in range(1, max_games + 1, chunk)]

        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = []
            for (a, b) in ranges:
                futs.append(ex.submit(
                    _arena_run_chunk, a, b,
                    eval_use_net, explore_use_net,
                    eval_net_path, explore_net_path, net_device,
                    out_dir,
                    cfg_eval_runtime, cfg_explore_runtime,
                    None, None  # Windows: do local inference
                ))

            with tqdm(total=max_games,
                      desc=f"ARENA (parallel x{workers}, {max_games} games)",
                      dynamic_ncols=True, leave=True, mininterval=0.2, file=TQDM_OUT) as pbar:
                for fut in as_completed(futs):
                    ew, xw, d, rows = fut.result()
                    eval_w += ew; ex_w += xw; dr += d
                    rows_all.extend(rows)
                    pbar.update(len(rows))

        rows_all.sort(key=lambda r: r[0])
        return eval_w, ex_w, dr, rows_all

    # ---------- Fallback: SPRT disabled, but no parallelism available ----------
    for g in tqdm(range(1, max_games + 1),
                  desc=f"ARENA ({max_games} games)",
                  dynamic_ncols=True, leave=True, mininterval=0.2, file=TQDM_OUT):
        eval_is_white = (g % 2 == 1)
        pgn, result, plycount, term = play_arena_game(
            game_idx=g,
            cfg_eval=cfg_eval_runtime,
            cfg_explore=cfg_explore_runtime,
            eval_backend=(pol_eval, val_eval),
            explore_backend=(pol_ex, val_ex),
            max_plies=ARENA_MAX_PLIES,
            eval_is_white=eval_is_white,
            open_policy_plies=OPEN_POLICY_PLIES_ARENA,
            quickdraw_halfmoves=ARENA_QUICKDRAW_HALFMOVES
        )
        with open(os.path.join(out_dir, f"arena_game_{g}.pgn"), "w", encoding="utf-8") as f:
            f.write(pgn)

        if result == "win":
            eval_w += 1
        elif result == "loss":
            ex_w += 1
        else:
            dr += 1

        rows_all.append((g, "white" if eval_is_white else "black", plycount, term, result))

    return eval_w, ex_w, dr, rows_all

# -------------------- Chunked EXPLORE worker (for spawn/fork) --------------------
def _run_explore_chunk(start_idx: int, end_idx: int, modes: List[str],
                       base_seed_for_cycle: int,
                       use_net_explore: bool,
                       cfg_explore_runtime: dict,
                       req=None, resq=None) -> Dict[int, Tuple[str, Optional[str], Optional[str]]]:
    if CENTRAL_INFERENCE and (req is not None) and (resq is not None):
        _set_central_client(req, resq)

    cfg_obj = SearchConfig(**cfg_explore_runtime)

    out: Dict[int, Tuple[str, Optional[str]]] = {}
    for g in range(start_idx, end_idx + 1):
        mode_g = modes[g - 1]
        g_idx, pgn_text, pgn_path, data_path = run_one_game(
            g, base_seed_for_cycle, cfg_obj, "EXPLORE", mode_g, MAX_PLIES, DATA_DIR, SAVE_EXPLORE_DATASET,
            use_net_explore, EXPLORE_WEIGHTS_PATH, NET_DEVICE,
            1.0, EXPLORE_POLICY_EPS, OPEN_POLICY_PLIES_EXPLORE,
            quickdraw_halfmoves=EXPLORE_QUICKDRAW_HALFMOVES
        )
        out[g_idx] = (pgn_text, pgn_path, data_path)
    return out

# ------------------------------ MAIN ------------------------------
_CFG_EXPLORE_RUNTIME = CFG_EXPLORE

if __name__ == "__main__":
    setup_live_log(LOG_DIR, prefix="run", fsync_every=200)
    if IS_WIN:
        mp.freeze_support()
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    use_net_explore = USE_NET_EXPLORE
    for cycle in range(1, CYCLES + 1):
        CURRENT_CYCLE = cycle
        DRAW_PENALTY = get_draw_penalty(CURRENT_CYCLE)
        print(f"\n========== CYCLE {cycle}/{CYCLES} ==========")
        cycle_tag = f"_c{cycle}" if CYCLES > 1 else ""

        # --- NEW: safety if baseline is missing ---
        if use_net_explore and not os.path.exists(EXPLORE_WEIGHTS_PATH):
            print("[warn] explore baseline weights not found; using heuristic exploration this cycle.")
            use_net_explore = False

        _CFG_EXPLORE_RUNTIME = SearchConfig(**{**CFG_EXPLORE.__dict__})
        alpha_c, eps_c = explore_dirichlet_schedule(cycle)
        _CFG_EXPLORE_RUNTIME.root_dirichlet_alpha = alpha_c
        _CFG_EXPLORE_RUNTIME.root_dirichlet_eps   = eps_c
        print(f"[info] EXPLORE root noise this cycle: alpha={alpha_c:.4f}, eps={eps_c:.4f}")
        if cycle <= 12:
            _CFG_EXPLORE_RUNTIME.topk_expand = 64
        else:
            _CFG_EXPLORE_RUNTIME.topk_expand = CFG_EXPLORE.topk_expand

        EXPLORE_DATASET_PATH = os.path.join(DATA_DIR, f"dataset_explore_{EXPLORE_GAMES}{cycle_tag}.npz")
        EVAL_DATASET_PATH    = os.path.join(DATA_DIR, f"dataset_eval_{EVAL_GAMES}{cycle_tag}.npz")
        explore_combined_path = os.path.join(EXPLORE_DIR, f"explore_all_{EXPLORE_GAMES}{cycle_tag}.pgn")
        eval_combined_path    = os.path.join(EVAL_DIR, f"eval_all_{EVAL_GAMES}{cycle_tag}.pgn")
        arena_csv_path        = os.path.join(ARENA_DIR, "arena_results.csv")
        arena_detail_path     = os.path.join(ARENA_DIR, f"arena_details{cycle_tag}.csv")
        arena_combined_path   = None
        manifest_path         = os.path.join(OUT_DIR, f"manifest{cycle_tag}.json")

        # --- NEW: mini-arena (net vs heuristic) paths ---
        miniarena_dir         = os.path.join(ARENA_DIR, "vs_heur")
        os.makedirs(miniarena_dir, exist_ok=True)
        miniarena_csv_path    = os.path.join(miniarena_dir, "arena_vs_heur_results.csv")
        miniarena_detail_path = os.path.join(miniarena_dir, f"arena_vs_heur_details{cycle_tag}.csv")

        manifest = {
            "cycle": cycle,
            "seeds": {"base": BASE_SEED},
            "device": NET_DEVICE,
            "central_inference": CENTRAL_INFERENCE,
            "explore_games": EXPLORE_GAMES,
            "eval_games": EVAL_GAMES,
            "arena_games": ARENA_GAMES,
            "cfg_explore": vars(_CFG_EXPLORE_RUNTIME),
            "cfg_eval":    vars(CFG),
            "cfg_arena":   vars(CFG_ARENA),
            "quickdraw": {
                "explore": EXPLORE_QUICKDRAW_HALFMOVES,
                "eval": EVAL_QUICKDRAW_HALFMOVES,
                "arena": ARENA_QUICKDRAW_HALFMOVES
            },
            "sprt": {"use": SPRT_USE, "p0": SPRT_P0, "p1": SPRT_P1, "alpha": SPRT_ALPHA, "beta": SPRT_BETA}
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        print(f"[info] manifest -> {manifest_path}")

        # --- NEW: per-cycle ARENA runtime config (slight noise + relaxed top-k + firmer early-stop) ---
        _CFG_ARENA_RUNTIME = SearchConfig(**{**CFG_ARENA.__dict__})
        if cycle <= 12:
            _CFG_ARENA_RUNTIME.topk_expand = 64
        else:
            _CFG_ARENA_RUNTIME.topk_expand = CFG_ARENA.topk_expand
        _CFG_ARENA_RUNTIME.root_dirichlet_eps = 0.04
        _CFG_ARENA_RUNTIME.early_stop_min_visits = max(CFG_ARENA.early_stop_min_visits, 48)
        _CFG_ARENA_RUNTIME.early_stop_best_ratio = 0.70

        # --- NEW: per-cycle EVAL runtime config (only relax top-k early) ---
        _CFG_EVAL_RUNTIME = SearchConfig(**{**CFG.__dict__})
        if cycle <= 12:
            _CFG_EVAL_RUNTIME.topk_expand = 64
        else:
            _CFG_EVAL_RUNTIME.topk_expand = CFG.topk_expand

        if CENTRAL_INFERENCE and use_net_explore:
            _start_central_server(EXPLORE_WEIGHTS_PATH, NET_DEVICE)
            _set_central_client(_CENTRAL_REQ, _CENTRAL_RESQ)

        print(f"[info] EXPLORE start @ {datetime.datetime.now().strftime('%H:%M:%S')}")
        explore_pairs: Dict[int, Tuple[str, Optional[str], Optional[str]]] = {}
        explore_modes: List[str] = []

        frac = EXPLORE_MCTS_FRACTION_BASE + 0.08 * (cycle - 1) + (0.12 if use_net_explore else 0.0)
        frac = float(max(0.10, min(0.80, frac)))
        mcts_games = int(round(EXPLORE_GAMES * frac))
        policy_games = EXPLORE_GAMES - mcts_games
        explore_modes.extend(["mcts"] * mcts_games + ["policy"] * policy_games)
        random.Random(BASE_SEED + cycle*12345).shuffle(explore_modes)

        t_start_explore = time.time()
        base_seed_for_cycle = BASE_SEED + (cycle-1)*100000

        # -------------------- Updated parallel EXPLORE (per-game progress) --------------------
        explore_parallel_used = False
        if PARALLEL_EXPLORE and EXPLORE_GAMES > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            ctx_name = "spawn" if os.name == "nt" else "fork"
            ctx = mp.get_context(ctx_name)
            try:
                # Force per-game tasks so tqdm ticks 1/405, 2/405, ... even in parallel
                CHUNK_SIZE = 1 if PER_GAME_PROGRESS else (EXPLORE_GAMES + MAX_WORKERS - 1) // MAX_WORKERS
                ranges = [(s, min(EXPLORE_GAMES, s + CHUNK_SIZE - 1))
                          for s in range(1, EXPLORE_GAMES + 1, CHUNK_SIZE)]

                print(f"[info] EXPLORE parallelization attempt: workers={MAX_WORKERS}, chunk_size={CHUNK_SIZE}")
                with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
                    explore_parallel_used = True
                    futs = []
                    for (a,b) in ranges:
                        futs.append(ex.submit(
                            _run_explore_chunk, a, b, explore_modes,
                            base_seed_for_cycle, use_net_explore,
                            _CFG_EXPLORE_RUNTIME.__dict__,
                            None, None  # Windows: don't pass mp.Queues; workers do local inference
                        ))
                    title = f"EXPLORE (parallel x{MAX_WORKERS}, per-game)"
                    with tqdm(total=EXPLORE_GAMES,
                              desc=title,
                              mininterval=0.2, dynamic_ncols=True, leave=True, file=TQDM_OUT) as pbar:
                        for fut in as_completed(futs):
                            res_dict = fut.result()
                            explore_pairs.update(res_dict)
                            pbar.update(len(res_dict))  # len==1 per future when CHUNK_SIZE=1
                print(f"[info] EXPLORE parallelization succeeded with {MAX_WORKERS} workers.")
            except OSError as e:
                print(f"[warn] Falling back to sequential due to spawn/fork error: {e}")
                PARALLEL_EXPLORE = False
                explore_parallel_used = False

        if (not PARALLEL_EXPLORE) or EXPLORE_GAMES <= 1 or (not explore_parallel_used):
            if CENTRAL_INFERENCE and use_net_explore:
                _set_central_client(_CENTRAL_REQ, _CENTRAL_RESQ)
            print("[info] EXPLORE running sequentially.")
            for g in tqdm(range(1, EXPLORE_GAMES + 1),
                          desc=f"EXPLORE (sequential {EXPLORE_GAMES})",
                          mininterval=0.2, dynamic_ncols=True, leave=True, file=TQDM_OUT):
                mode_g = explore_modes[g - 1]
                g_idx, pgn_text, pgn_path, data_path = run_one_game(
                    g, base_seed_for_cycle, _CFG_EXPLORE_RUNTIME, "EXPLORE", mode_g, MAX_PLIES, DATA_DIR, SAVE_EXPLORE_DATASET,
                    use_net_explore, EXPLORE_WEIGHTS_PATH, NET_DEVICE,
                    1.0, EXPLORE_POLICY_EPS, OPEN_POLICY_PLIES_EXPLORE,
                    quickdraw_halfmoves=EXPLORE_QUICKDRAW_HALFMOVES
                )
                explore_pairs[g_idx] = (pgn_text, pgn_path, data_path)

        print(f"[info] EXPLORE done in {time.time()-t_start_explore:.1f}s")

        with open(explore_combined_path, "w", encoding="utf-8") as f:
            for g in range(1, EXPLORE_GAMES + 1):
                pgn_text, pgn_path, _ = explore_pairs[g]
                if pgn_text:
                    f.write(pgn_text + "\n")
                elif pgn_path and os.path.exists(pgn_path):
                    with open(pgn_path, "r", encoding="utf-8") as pg:
                        f.write(pg.read() + "\n")

        if SAVE_EXPLORE_DATASET:
            explore_data_paths = [explore_pairs[g][2] for g in range(1, EXPLORE_GAMES + 1)]
            ok_concat = concat_npz_sparse(explore_data_paths, EXPLORE_DATASET_PATH)
        else:
            ok_concat = False

        USE_NET_EVAL = False
        if TRAIN_AFTER_EXPLORE and SAVE_EXPLORE_DATASET and ok_concat:
            # >>> NEW: build replay buffer (oldest->newest) if available
            replay_paths = []
            start_c = max(1, cycle - REPLAY_K + 1)
            for c in range(start_c, cycle + 1):
                p = os.path.join(DATA_DIR, f"dataset_explore_{EXPLORE_GAMES}_c{c}.npz")
                if os.path.exists(p):
                    replay_paths.append(p)
            TRAIN_SOURCE_PATH = EXPLORE_DATASET_PATH
            if len(replay_paths) >= 2:
                REPLAY_PATH = os.path.join(DATA_DIR, f"dataset_replay_k{REPLAY_K}_c{cycle}.npz")
                ok_replay = concat_npz_sparse(replay_paths, REPLAY_PATH)
                if ok_replay:
                    TRAIN_SOURCE_PATH = REPLAY_PATH
                    print(f"[train] using replay buffer with {len(replay_paths)} cycles -> {REPLAY_PATH}")
                else:
                    print("[train] replay concat failed; falling back to current cycle dataset.")

            print(f"[info] TRAIN start @ {datetime.datetime.now().strftime('%H:%M:%S')}")
            # --- Dynamic epoch count based on dataset size (target ~1200 optimizer steps) ---
            try:
                with np.load(TRAIN_SOURCE_PATH) as _tmpd:
                    _N = int(_tmpd["X"].shape[0])
            except Exception:
                _N = 0
            _steps_per_epoch = max(1, int(math.ceil(_N / max(1, BATCH_SIZE))))
            _target_steps = 1200
            _epochs_used = int(math.ceil(_target_steps / _steps_per_epoch)) if _N > 0 else EPOCHS
            _epochs_used = int(max(2, min(6, _epochs_used)))
            print(f"[train] dataset N={_N} | steps/epoch≈{_steps_per_epoch} | epochs={_epochs_used}")

            ok = _train_on_npz(
                npz_path=TRAIN_SOURCE_PATH,
                out_weights_path=CANDIDATE_WEIGHTS_PATH,
                epochs=_epochs_used,
                batch_size=BATCH_SIZE,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                value_loss_weight=VALUE_LOSS_WEIGHT,
                device=NET_DEVICE,
            )
            USE_NET_EVAL = ok
            if not ok:
                print("[warn] Training skipped/failed; evaluating with heuristics.")
        else:
            print("[info] Skipping training (no dataset or disabled).")

        if CENTRAL_INFERENCE and USE_NET_EVAL:
            _start_central_server(NET_WEIGHTS_PATH, NET_DEVICE)
            _set_central_client(_CENTRAL_REQ, _CENTRAL_RESQ)

        if USE_NET_EVAL:
            print(f"[info] ARENA start @ {datetime.datetime.now().strftime('%H:%M:%S')} | max_games={ARENA_GAMES} | SPRT={SPRT_USE}")
            eval_w, ex_w, dr, rows = run_arena_sprt(
                max_games=ARENA_GAMES,
                eval_use_net=True,
                explore_use_net=use_net_explore,
                eval_net_path=CANDIDATE_WEIGHTS_PATH,
                explore_net_path=EXPLORE_WEIGHTS_PATH,
                net_device=NET_DEVICE,
                out_dir=ARENA_DIR,
                cfg_eval_runtime=_CFG_ARENA_RUNTIME,
                cfg_explore_runtime=_CFG_ARENA_RUNTIME
            )
            n = eval_w + ex_w + dr
            score = _score_from_wld(eval_w, dr, ex_w)
            elo   = _elo_from_score(score)
            delta_elo, _ = _elo_ci(score, max(1,n))
            print(f"[arena] eval_wins={eval_w} | explore_wins={ex_w} | draws={dr} "
                  f"| score={score:.3f} | elo_vs_explore={elo:.1f} ± {delta_elo:.1f} (95% CI)")

            newfile = not os.path.exists(arena_csv_path)
            with open(arena_csv_path, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if newfile:
                    w.writerow(["cycle","timestamp","games","eval_wins","explore_wins","draws","score","elo","elo_ci95"])
                w.writerow([cycle, datetime.datetime.now().isoformat(timespec='seconds'),
                            n, eval_w, ex_w, dr, f"{score:.4f}", f"{elo:.1f}", f"{delta_elo:.1f}"])

            with open(arena_detail_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["game","eval_color","plycount","termination","result_eval_pov"])
                w.writerows(rows)

            arena_combined_path = os.path.join(ARENA_DIR, f"arena_all_{n}{cycle_tag}.pgn")
            with open(arena_combined_path, "w", encoding="utf-8") as f:
                for g in range(1, n + 1):
                    with open(os.path.join(ARENA_DIR, f"arena_game_{g}.pgn"), "r", encoding="utf-8") as pg:
                        f.write(pg.read() + "\n")

            promote = False
            if SPRT_USE:
                d = sprt_decision(eval_w, dr, ex_w)
                if d == "accept":
                    promote = True
            else:
                eval_pts = eval_w + 0.5*dr
                ex_pts   = ex_w + 0.5*dr
                promote  = (eval_pts > ex_pts)

            if promote:
                print("[arena] PROMOTION: evaluation becomes the new exploration policy.")
                # --- NEW: copy candidate -> baseline ---
                import shutil
                try:
                    shutil.copy2(CANDIDATE_WEIGHTS_PATH, EXPLORE_WEIGHTS_PATH)
                    ema_src = CANDIDATE_WEIGHTS_PATH + ".ema"
                    if os.path.exists(ema_src):
                        shutil.copy2(ema_src, EXPLORE_WEIGHTS_PATH + ".ema")
                except Exception as e:
                    print(f"[warn] failed to update baseline: {e}")
                use_net_explore = True
            else:
                print("[arena] No promotion this cycle. Exploration policy stays as-is.")
            # --- NEW: Mini-arena: Net vs Heuristic (20–30 games) ---
            print(f"[info] MINI-ARENA (net vs heuristic) start @ {datetime.datetime.now().strftime('%H:%M:%S')} | games={MINIARENA_GAMES}")
            eval_w2, ex_w2, dr2, rows2 = run_arena_sprt(
                max_games=MINIARENA_GAMES,
                eval_use_net=True,
                explore_use_net=False,  # heuristic opponent
                eval_net_path=CANDIDATE_WEIGHTS_PATH,
                explore_net_path=EXPLORE_WEIGHTS_PATH,   # (ignored since heuristic)
                net_device=NET_DEVICE,
                out_dir=miniarena_dir,  # avoids PGN filename collisions
                cfg_eval_runtime=_CFG_ARENA_RUNTIME,
                cfg_explore_runtime=_CFG_ARENA_RUNTIME
            )
            n2 = eval_w2 + ex_w2 + dr2
            score2 = _score_from_wld(eval_w2, dr2, ex_w2)
            elo2   = _elo_from_score(score2)
            delta_elo2, _ = _elo_ci(score2, max(1, n2))
            print(f"[mini-arena] vs-heur | W={eval_w2} D={dr2} L={ex_w2} | score={score2:.3f} | elo={elo2:.1f} ± {delta_elo2:.1f} (95% CI)")

            newfile2 = not os.path.exists(miniarena_csv_path)
            with open(miniarena_csv_path, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if newfile2:
                    w.writerow(["cycle","timestamp","games","net_wins","heur_wins","draws","score","elo","elo_ci95"])
                w.writerow([cycle, datetime.datetime.now().isoformat(timespec='seconds'),
                            n2, eval_w2, ex_w2, dr2, f"{score2:.4f}", f"{elo2:.1f}", f"{delta_elo2:.1f}"])

            with open(miniarena_detail_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["game","net_color","plycount","termination","result_net_pov"])
                w.writerows(rows2)
            
        else:
            print("[info] ARENA skipped (no trained eval net).")

        print(f"[info] EVAL start @ {datetime.datetime.now().strftime('%H:%M:%S')}")
        t_start_eval = time.time()
        eval_pairs: Dict[int, Tuple[str, Optional[str]]] = {}
        if CENTRAL_INFERENCE and USE_NET_EVAL:
            _start_central_server(CANDIDATE_WEIGHTS_PATH, NET_DEVICE)  # <-- CHANGED
            _set_central_client(_CENTRAL_REQ, _CENTRAL_RESQ)
        for g in tqdm(range(1, EVAL_GAMES + 1),
                      desc=f"EVAL (sequential {EVAL_GAMES})",
                      mininterval=0.2, dynamic_ncols=True, leave=True, file=TQDM_OUT):
            g_idx, pgn_text, pgn_path, data_path = run_one_game(
                g, base_seed_for_cycle, _CFG_EVAL_RUNTIME, "EVAL", "mcts", MAX_PLIES, DATA_DIR, SAVE_EVAL_DATASET,
                USE_NET_EVAL, CANDIDATE_WEIGHTS_PATH, NET_DEVICE,   # <-- CHANGED
                open_policy_plies=OPEN_POLICY_PLIES_EVAL,
                quickdraw_halfmoves=EVAL_QUICKDRAW_HALFMOVES
            )
            eval_pairs[g_idx] = (pgn_text, pgn_path, data_path)
        print(f"[info] EVAL done in {time.time()-t_start_eval:.1f}s")

        with open(eval_combined_path, "w", encoding="utf-8") as f:
            for g in range(1, EVAL_GAMES + 1):
                pgn_text, pgn_path, _ = eval_pairs[g]
                if pgn_text:
                    f.write(pgn_text + "\n")
                elif pgn_path and os.path.exists(pgn_path):
                    with open(pgn_path, "r", encoding="utf-8") as pg:
                        f.write(pg.read() + "\n")

        if SAVE_EVAL_DATASET:
            eval_data_paths = [eval_pairs[g][2] for g in range(1, EVAL_GAMES + 1)]
            concat_npz_sparse(eval_data_paths, EVAL_DATASET_PATH)

        print(f"\n[summary] CYCLE {cycle}:")
        print(f"  Explore PGN: {explore_combined_path}")
        if arena_combined_path:
            print(f"  Arena PGN:   {arena_combined_path}")
            print(f"  Arena details CSV: {arena_detail_path}")
        print(f"  Eval PGN:    {eval_combined_path}")
        if SAVE_EXPLORE_DATASET:
            print(f"  Explore dataset: {EXPLORE_DATASET_PATH}")
        if SAVE_EVAL_DATASET:
            print(f"  Eval dataset:    {EVAL_DATASET_PATH}")
        print(f"  Exploration backend for NEXT cycle: {'NET' if use_net_explore else 'HEURISTIC'}")

    if CENTRAL_INFERENCE:
        _stop_central_server()

    print("\n[done] All cycles complete.")