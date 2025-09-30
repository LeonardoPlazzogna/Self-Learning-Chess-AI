from typing import List, Tuple, Optional, Dict, Set, Hashable
import numpy as np

# -----------------------------------------------------------------------------
# Chess Environment (8x8 board, 0-based indices) + Zobrist TT & Shared Stats
# -----------------------------------------------------------------------------
# Board encoding (integers):
#   0  : empty
#   +1 : white pawn       -1 : black pawn
#   +2 : white knight     -2 : black knight
#   +3 : white bishop     -3 : black bishop
#   +4 : white rook       -4 : black rook
#   +5 : white queen      -5 : black queen
#   +6 : white king       -6 : black king
#
# Coordinate system:
#   - Rows r in [0..7] from White's back rank (r=0) toward Black's back rank (r=7).
#   - Columns c in [0..7] from queenside (c=0) to kingside (c=7).
#   - White pawns move "down" the array (increasing row index, direction +1).
#
# Moves are represented as 4-tuples: (from_row, from_col, to_row, to_col)
# Pseudo-legal moves ignore self-check; legal moves filter out those that leave
# the moving side in check.
#
# Game rules covered:
#   - Normal piece moves and captures
#   - En passant (stateful, valid exactly for the immediate reply)
#   - Castling (with full path/attack checks and rook hop)
#   - Auto-queen promotion on the last rank
#   - Fifty-move rule (automatic draw at 100 half-moves with no pawn move/capture)
#   - Threefold repetition (automatic draw when the same position occurs 3 times)
#   - Draw by insufficient material (dead positions)
#
# Zobrist Transposition Table:
#   - env.zkey maintained incrementally in push()/pop()
#   - Hash includes: piece-square table, side-to-move, castling rights, and
#     a canonical en-passant FILE (only if capture is possible for side-to-move)
#   - env.tt[zkey] stores shared stats: {N, W, Q, v_net, P(policy cache),
#     virtual_loss} and can be used by your MCTS/PUCT. The same child state,
#     reached via different parents (transpositions), shares the same entry.
#   - Utility helpers provided: tt_get, tt_set_policy, tt_add_vloss, tt_sub_vloss
#     (these do not implement a full search; they’re primitives for your loop).
# -----------------------------------------------------------------------------

Move = Tuple[int, int, int, int]
Board = List[List[int]]

# ---------------------- FACTORED POLICY DELTA VOCAB (D=73) ----------------------
# White-centric canonical deltas. Black moves are mapped by 180° rotation
# so that "forward" is +row and left/right match white's perspective.
# Layout: first 56 = queen-like slides (8 dirs × steps 1..7),
#         then 8 knight jumps,
#         then 9 underpromotion placeholders: (R,B,N) × (F, L, R).
_DIRS_8 = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]

def _build_delta_table_73() -> List[Tuple[int,int,str,Tuple]]:
    table: List[Tuple[int,int,str,Tuple]] = []
    # 56 sliding (Queen-like)
    for (dr, dc) in _DIRS_8:
        for k in range(1, 8):
            table.append((dr * k, dc * k, "SLIDE", (dr, dc, k)))
    # 8 knight
    for (dr, dc) in [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]:
        table.append((dr, dc, "KNIGHT", ()))
    # 9 underpromotion placeholders: (piece in {R,B,N}) × (F,L,R)
    # Geometry uses dr=+1 and dc ∈ {0,-1,+1} to match the last-rank step.
    for promo in ("R","B","N"):
        for cap in ("F","L","R"):  # forward, capture-left, capture-right
            dc = 0 if cap=="F" else (-1 if cap=="L" else +1)
            table.append((1, dc, f"UNDER_{promo}_{cap}", (promo, cap)))
    return table

DELTA_TABLE_73: List[Tuple[int,int,str,Tuple]] = _build_delta_table_73()
D73: int = len(DELTA_TABLE_73)  # 73
# Fast lookup for slide/knight by (dr,dc). Underpromotions will not be looked up here.
DELTA_TO_INDEX_73: Dict[Tuple[int,int], int] = {
    (dr, dc): idx for idx, (dr, dc, kind, meta) in enumerate(DELTA_TABLE_73)
    if kind in ("SLIDE","KNIGHT")
}

def _rot180_sq(index64: int) -> int:
    """Rotate a square index by 180° (white<->black perspective)."""
    r, c = divmod(index64, 8)
    return (7 - r) * 8 + (7 - c)

def _rot180_rc(r: int, c: int) -> Tuple[int,int]:
    return (7 - r, 7 - c)

# -----------------------------------------------------------------------------
# Exception hierarchy and safety constants (non-intrusive)
# -----------------------------------------------------------------------------
class ChessEnvError(Exception):
    """Base class for ChessEnv-specific errors."""

class GameAlreadyFinishedError(ChessEnvError):
    pass

class InvalidMoveFormatError(ChessEnvError):
    pass

class OutOfBoundsError(ChessEnvError):
    pass

class EmptySquareError(ChessEnvError):
    pass

class WrongPieceColorError(ChessEnvError):
    pass

class NotYourTurnError(ChessEnvError):
    pass

class IllegalMoveError(ChessEnvError):
    pass

class InvalidStateError(ChessEnvError):
    pass

class MultipleKingsError(InvalidStateError):
    pass

# Allowed piece codes for validation
_ALLOWED_PIECES: Set[int] = set(range(-6, 7))


class ChessEnv:
    """Minimal, self-contained chess environment with legal move generation + TT."""

    # ---------------------------- ZOBRIST SETUP --------------------------------
    # We prebuild a deterministic PRNG and fill Zobrist tables on first instance.
    _ZOBRIST_INIT_DONE = False
    _Z_PIECE: Optional[List[List[int]]] = None    # 12 x 64 [piece_index][sq]
    _Z_CASTLING: Optional[List[int]] = None       # [K,Q,k,q] order
    _Z_SIDE: Optional[int] = None                 # side to move
    _Z_EP_FILE: Optional[List[int]] = None        # 8 ep files (a..h)

    @staticmethod
    def _seeded_rand64(seed: int) -> int:
        # xorshift64* deterministic generator
        x = seed & ((1 << 64) - 1)
        x ^= (x >> 12) & ((1 << 64) - 1)
        x ^= (x << 25) & ((1 << 64) - 1)
        x ^= (x >> 27) & ((1 << 64) - 1)
        return (x * 2685821657736338717) & ((1 << 64) - 1)

    @classmethod
    def _init_zobrist_tables(cls) -> None:
        if cls._ZOBRIST_INIT_DONE:
            return
        # Base seed for reproducibility
        seed = 0x9E3779B97F4A7C15
        def nxt():
            nonlocal seed
            seed = cls._seeded_rand64(seed)
            return seed

        # piece indexing: map piece code -> index [0..11]
        # order: White: P,N,B,R,Q,K  Black: p,n,b,r,q,k
        cls._PIECE_TO_INDEX = {
            1:0, 2:1, 3:2, 4:3, 5:4, 6:5,
           -1:6,-2:7,-3:8,-4:9,-5:10,-6:11
        }

        # 12 x 64 Zobrist piece table
        cls._Z_PIECE = [[nxt() for _ in range(64)] for _ in range(12)]

        # castling K,Q,k,q
        cls._Z_CASTLING = [nxt(), nxt(), nxt(), nxt()]

        # side to move
        cls._Z_SIDE = nxt()

        # en-passant by file a..h
        cls._Z_EP_FILE = [nxt() for _ in range(8)]

        cls._ZOBRIST_INIT_DONE = True

    # -------------------------------------------------------------------------
    # Attributes
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialize and reset the environment."""
        self._init_zobrist_tables()

        # Transposition table: zkey -> entry
        # Entry schema:
        #   {
        #     "N": int, "W": float, "Q": float,
        #     "v_net": Optional[float],
        #     "P": Optional[Dict[Move,float]],    # policy cache (move->prior)
        #     "virtual_loss": int
        #   }
        self.tt: Dict[int, Dict] = {}

        self.reset()
        # Reversible history stack for push/pop (used by search)
        self._history: List[dict] = []
        self.info: Dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def reset(self) -> Board:
        """Reset to the initial chess position and return the board."""
        self.board = self._initial_board()
        self.current_player = 1  # +1 white, -1 black
        self.done = False
        self.en_passant_square: Optional[Tuple[int,int]] = None
        self.castling_rights = {"K": True, "Q": True, "k": True, "q": True}
        self.halfmove_clock = 0
        self.position_counts: Dict[Hashable, int] = {}

        # Build initial Zobrist key from scratch
        self.zkey: int = self._compute_zobrist_key()

        self._increment_position_repetition()
        self._validate_full_state(strict=True)
        return self.board

    def _initial_board(self) -> Board:
        """Create the standard chess initial position."""
        board: Board = [[0] * 8 for _ in range(8)]
        for i in range(8):
            board[1][i] = 1
            board[6][i] = -1
        pieces = [4, 2, 3, 5, 6, 3, 2, 4]
        for i, p in enumerate(pieces):
            board[0][i] = p
            board[7][i] = -p
        return board

    # ---------------------- Zobrist helpers (public-ish) -----------------------
    def tt_get(self) -> Dict:
        """Get (or create) the TT entry for current env.zkey."""
        entry = self.tt.get(self.zkey)
        if entry is None:
            entry = {"N": 0, "W": 0.0, "Q": 0.0, "v_net": None, "P": None, "virtual_loss": 0}
            self.tt[self.zkey] = entry
        return entry

    def tt_peek(self, zkey: int) -> Optional[Dict]:
        """Peek an entry by zkey (no create)."""
        return self.tt.get(zkey)

    def tt_set_policy(self, move_priors: Dict[Move, float]) -> None:
        """Cache policy for the current state (overwrite)."""
        entry = self.tt_get()
        entry["P"] = dict(move_priors)

    def tt_add_vloss(self, amount: int = 1) -> None:
        """Add virtual loss to current node (useful in parallel MCTS)."""
        entry = self.tt_get()
        entry["virtual_loss"] += amount
        # A common pattern is also to tentatively increment N to discourage visits
        entry["N"] += amount

    def tt_sub_vloss(self, amount: int = 1) -> None:
        """Remove virtual loss (on backup)."""
        entry = self.tt_get()
        entry["virtual_loss"] = max(0, entry["virtual_loss"] - amount)
        entry["N"] = max(0, entry["N"] - amount)

    # -------------------------------------------------------------------------
    # Push/Pop for fast MCTS (incremental Zobrist updates)
    # -------------------------------------------------------------------------
    def _piece_index(self, piece: int) -> int:
        idx = self._PIECE_TO_INDEX.get(piece)  # type: ignore
        if idx is None:
            raise InvalidStateError(f"Bad piece code for zobrist: {piece}")
        return idx

    def _sq_index(self, r: int, c: int) -> int:
        return r * 8 + c

    def _xor_piece(self, piece: int, r: int, c: int) -> None:
        if piece == 0:
            return
        idx = self._piece_index(piece)
        sq = self._sq_index(r, c)
        self.zkey ^= self._Z_PIECE[idx][sq]  # type: ignore

    def _xor_castling_bits(self, before: Dict[str,bool], after: Dict[str,bool]) -> None:
        # XOR in/out differences for K,Q,k,q bit toggles
        keys = ["K","Q","k","q"]
        for i,k in enumerate(keys):
            if before[k] != after[k]:
                self.zkey ^= self._Z_CASTLING[i]  # type: ignore

    def _xor_side(self) -> None:
        self.zkey ^= self._Z_SIDE  # type: ignore

    def _canonical_ep_file_for_hash(self, side_to_move: int) -> Optional[int]:
        """Return ep FILE (0..7) if a capture is possible by side_to_move, else None."""
        ep = self.en_passant_square
        if ep is None:
            return None
        ep_r, ep_c = ep
        # For hashing, include EP file only if *current* side to move can capture onto it.
        dr = -1 if side_to_move > 0 else 1  # because to capture onto ep square, pawn must be one rank behind ep square
        for dc in (-1, 1):
            pr, pc = ep_r + dr, ep_c + dc
            if self.in_bounds(pr, pc) and self.board[pr][pc] == 1 * side_to_move:
                return ep_c  # file index
        return None

    def _xor_ep_if_any(self, side_to_move: int, add: bool) -> None:
        """XOR EP file key in/out depending on side_to_move."""
        ep_file = self._canonical_ep_file_for_hash(side_to_move)
        if ep_file is not None:
            self.zkey ^= self._Z_EP_FILE[ep_file]  # type: ignore

    def _compute_zobrist_key(self) -> int:
        """Compute fresh Zobrist key from current full state."""
        z = 0
        # pieces
        for r in range(8):
            for c in range(8):
                p = self.board[r][c]
                if p != 0:
                    idx = self._piece_index(p)
                    z ^= self._Z_PIECE[idx][self._sq_index(r,c)]  # type: ignore
        # castling
        keys = ["K","Q","k","q"]
        for i,k in enumerate(keys):
            if self.castling_rights[k]:
                z ^= self._Z_CASTLING[i]  # type: ignore
        # side
        if self.current_player == -1:
            z ^= self._Z_SIDE  # type: ignore
        # en-passant (canonical)
        ep_file = self._canonical_ep_file_for_hash(self.current_player)
        if ep_file is not None:
            z ^= self._Z_EP_FILE[ep_file]  # type: ignore
        return z

    def push(self, move: Move) -> dict:
        """Apply a move and store a reversible record (for search). Zobrist-safe."""
        fr, fc, tr, tc = move
        piece = self.board[fr][fc]
        target = self.board[tr][tc]

        # record previous state for pop
        record = {
            "move": move,
            "piece": piece,
            "captured": target,
            "en_passant": self.en_passant_square,
            "castling_rights": dict(self.castling_rights),
            "halfmove_clock": self.halfmove_clock,
            "current_player": self.current_player,
            "done": self.done,
            "zkey": self.zkey,
        }

        # --- ZOBRIST: remove old EP (based on current side-to-move) BEFORE changes
        self._xor_ep_if_any(self.current_player, add=False)

        is_capture = target != 0
        is_pawn_move = abs(piece) == 1

        # en passant capture
        ep_captured = None
        if abs(piece) == 1 and (tr, tc) == self.en_passant_square and target == 0 and fc != tc:
            ep_r, ep_c = fr, tc
            ep_captured = self.board[ep_r][ep_c]
            record["ep_captured"] = ep_captured
            # ZOBRIST: remove captured pawn at (ep_r,ep_c)
            self._xor_piece(ep_captured, ep_r, ep_c)
            self.board[ep_r][ep_c] = 0
            is_capture = True

        # ZOBRIST: move piece: toggle from source and destination (handle capture)
        self._xor_piece(piece, fr, fc)     # remove from source
        if target != 0:
            self._xor_piece(target, tr, tc)  # remove captured on target
        # apply move on board
        self.board[tr][tc] = piece
        self.board[fr][fc] = 0
        # ZOBRIST: add piece at destination
        self._xor_piece(piece, tr, tc)

        # castling rook hop
        rook_moved = None
        if abs(piece) == 6 and fr == tr and abs(tc - fc) == 2:
            if tc == fc + 2:
                rr_from, rr_to = (fr, 7), (fr, 5)
            else:
                rr_from, rr_to = (fr, 0), (fr, 3)
            rook = self.board[rr_from[0]][rr_from[1]]
            record["rook_move"] = (rr_from, rr_to, rook)
            # ZOBRIST: rook move
            self._xor_piece(rook, rr_from[0], rr_from[1])
            self._xor_piece(rook, rr_to[0], rr_to[1])
            self.board[rr_to[0]][rr_to[1]] = rook
            self.board[rr_from[0]][rr_from[1]] = 0
            rook_moved = True

        # promotion
        if abs(piece) == 1 and ((piece > 0 and tr == 7) or (piece < 0 and tr == 0)):
            # ZOBRIST: replace pawn with queen
            self._xor_piece(piece, tr, tc)  # remove pawn at dst
            q = 5 if piece > 0 else -5
            self.board[tr][tc] = q
            self._xor_piece(q, tr, tc)      # add queen
            record["promotion"] = True

        # en passant square update
        if abs(piece) == 1 and abs(tr - fr) == 2:
            self.en_passant_square = ((fr + tr) // 2, fc)
        else:
            self.en_passant_square = None

        # update castling rights (track Zobrist differences)
        before_cr = dict(self.castling_rights)
        if piece == 6:
            self.castling_rights["K"] = self.castling_rights["Q"] = False
        elif piece == -6:
            self.castling_rights["k"] = self.castling_rights["q"] = False
        if piece == 4:
            if (fr, fc) == (0, 0): self.castling_rights["Q"] = False
            if (fr, fc) == (0, 7): self.castling_rights["K"] = False
        elif piece == -4:
            if (fr, fc) == (7, 0): self.castling_rights["q"] = False
            if (fr, fc) == (7, 7): self.castling_rights["k"] = False
        if target == 4:
            if (tr, tc) == (0, 0): self.castling_rights["Q"] = False
            if (tr, tc) == (0, 7): self.castling_rights["K"] = False
        elif target == -4:
            if (tr, tc) == (7, 0): self.castling_rights["q"] = False
            if (tr, tc) == (7, 7): self.castling_rights["k"] = False
        self._xor_castling_bits(before_cr, self.castling_rights)

        # halfmove clock
        if is_capture or is_pawn_move:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # ZOBRIST: flip side to move
        self._xor_side()
        # ZOBRIST: add new EP (if any, canonical w.r.t. NEW side to move)
        self._xor_ep_if_any(self.current_player * -1, add=True)  # we already flipped, so new side is -current_player now

        # turn
        self.current_player *= -1

        self._history.append(record)
        return record

    def pop(self) -> None:
        """Undo the last move (for search), restoring Zobrist and state."""
        if not self._history:
            raise ChessEnvError("No moves to undo.")
        rec = self._history.pop()
        fr, fc, tr, tc = rec["move"]

        # Fully restore everything from record (including zkey)
        self.board[fr][fc] = rec["piece"]
        self.board[tr][tc] = rec["captured"]
        self.en_passant_square = rec["en_passant"]
        self.castling_rights = rec["castling_rights"]
        self.halfmove_clock = rec["halfmove_clock"]
        self.current_player = rec["current_player"]
        self.done = rec["done"]
        self.zkey = rec["zkey"]

        if "ep_captured" in rec:
            ep_r, ep_c = fr, tc
            self.board[ep_r][ep_c] = rec["ep_captured"]

        if "rook_move" in rec:
            (rr_from, rr_to, rook) = rec["rook_move"]
            self.board[rr_from[0]][rr_from[1]] = rook
            self.board[rr_to[0]][rr_to[1]] = 0

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def in_bounds(self, r: int, c: int) -> bool:
        """Return True if (r, c) lies within the 8x8 board."""
        return 0 <= r < 8 and 0 <= c < 8

    # -- Added: validation helpers ------------------------------------------------
    def _validate_board_matrix(self) -> None:
        if not isinstance(self.board, list) or len(self.board) != 8:
            raise InvalidStateError("Board must be a list of 8 rows.")
        for idx, row in enumerate(self.board):
            if not isinstance(row, list) or len(row) != 8:
                raise InvalidStateError(f"Row {idx} must be a list of length 8.")
            for jdx, v in enumerate(row):
                if not isinstance(v, int):
                    raise InvalidStateError(f"Board cell ({idx},{jdx}) must be int, got {type(v)}.")
                if v not in _ALLOWED_PIECES:
                    raise InvalidStateError(f"Invalid piece code {v} at ({idx},{jdx}).")

    def _validate_side(self) -> None:
        if self.current_player not in (1, -1):
            raise InvalidStateError(f"current_player must be +1 or -1, got {self.current_player}.")

    def _validate_kings(self) -> None:
        w, b = 0, 0
        w_pos, b_pos = None, None
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == 6:
                    w += 1; w_pos = (r, c)
                elif self.board[r][c] == -6:
                    b += 1; b_pos = (r, c)
        if w == 0 or b == 0:
            raise InvalidStateError(f"Missing king(s): white={w}, black={b}.")
        if w > 1 or b > 1:
            raise MultipleKingsError(f"Multiple kings detected: white={w} at {w_pos}, black={b} at {b_pos}.")

    def _validate_castling_rights(self) -> None:
        cr = self.castling_rights
        if not isinstance(cr, dict) or set(cr.keys()) != {"K", "Q", "k", "q"}:
            raise InvalidStateError("castling_rights must have keys {'K','Q','k','q'}.")
        for k, v in cr.items():
            if not isinstance(v, bool):
                raise InvalidStateError(f"Castling right {k} must be bool.")
        # Soft check
        if cr["K"]:
            if self.board[0][4] != 6 or self.board[0][7] != 4:
                raise InvalidStateError("Inconsistent white K-side rights with piece placement.")
        if cr["Q"]:
            if self.board[0][4] != 6 or self.board[0][0] != 4:
                raise InvalidStateError("Inconsistent white Q-side rights with piece placement.")
        if cr["k"]:
            if self.board[7][4] != -6 or self.board[7][7] != -4:
                raise InvalidStateError("Inconsistent black k-side rights with piece placement.")
        if cr["q"]:
            if self.board[7][4] != -6 or self.board[7][0] != -4:
                raise InvalidStateError("Inconsistent black q-side rights with piece placement.")

    def _validate_en_passant(self) -> None:
        ep = self.en_passant_square
        if ep is None:
            return
        if not (isinstance(ep, tuple) and len(ep) == 2 and all(isinstance(x, int) for x in ep)):
            raise InvalidStateError(f"en_passant_square must be (r,c) int tuple or None; got {ep}.")
        r, c = ep
        if not self.in_bounds(r, c):
            raise InvalidStateError(f"en_passant_square {ep} is out of bounds.")
        if r not in (2, 5):
            raise InvalidStateError(f"en_passant_square {ep} has invalid rank; expected row 2 or 5.")

    def _validate_halfmove_clock(self) -> None:
        if not isinstance(self.halfmove_clock, int) or self.halfmove_clock < 0:
            raise InvalidStateError(f"halfmove_clock must be a non-negative int, got {self.halfmove_clock}.")

    def _validate_full_state(self, strict: bool = False) -> None:
        self._validate_board_matrix()
        self._validate_side()
        self._validate_kings()
        self._validate_halfmove_clock()
        if strict:
            self._validate_castling_rights()
        self._validate_en_passant()

    # -- Position hashing helpers for repetition --------------------------------
    def _canonical_en_passant_for_key(self, side_to_move: int) -> Optional[Tuple[int, int]]:
        """Return the en-passant square *only if* a pawn of `side_to_move` can capture onto it."""
        ep = self.en_passant_square
        if ep is None:
            return None
        ep_r, ep_c = ep
        if not self.in_bounds(ep_r, ep_c):
            raise InvalidStateError(f"Stored en_passant_square {ep} is out of bounds.")
        dr = -1 if side_to_move > 0 else 1
        for dc in (-1, 1):
            pr, pc = ep_r + dr, ep_c + dc
            if self.in_bounds(pr, pc) and self.board[pr][pc] == 1 * side_to_move:
                return ep
        return None

    def _position_key(self) -> Hashable:
        """Build a canonical, hashable key of the *current* position for repetition."""
        flat: List[int] = []
        for r in range(8):
            for c in range(8):
                flat.append(self.board[r][c])

        cr = (self.castling_rights["K"], self.castling_rights["Q"],
              self.castling_rights["k"], self.castling_rights["q"])

        ep = self._canonical_en_passant_for_key(self.current_player)

        return (tuple(flat), self.current_player, cr, ep)

    def _increment_position_repetition(self) -> int:
        """Increment the repetition counter for the current position; return new count."""
        key = self._position_key()
        self.position_counts[key] = self.position_counts.get(key, 0) + 1
        return self.position_counts[key]

    # -------------------------------------------------------------------------
    # Move generation pipeline
    # -------------------------------------------------------------------------
    def pseudo_legal_moves(self, side: int) -> List[Move]:
        """Generate all pseudo-legal moves for `side` (ignores self-check)."""
        if side not in (1, -1):
            raise InvalidStateError(f"pseudo_legal_moves: side must be +1 or -1, got {side}.")
        self._validate_board_matrix()
        moves: List[Move] = []
        for r in range(8):
            for c in range(8):
                v = self.board[r][c]
                if v * side <= 0:
                    continue
                ap = abs(v)
                if ap == 1:
                    moves.extend(self._pawn_moves(r, c, v))
                elif ap == 2:
                    moves.extend(self._knight_moves(r, c, v))
                elif ap == 3:
                    moves.extend(self._bishop_moves(r, c, v))
                elif ap == 4:
                    moves.extend(self._rook_moves(r, c, v))
                elif ap == 5:
                    moves.extend(self._queen_moves(r, c, v))
                elif ap == 6:
                    moves.extend(self._king_moves(r, c, v))
                else:
                    raise InvalidStateError(f"Unknown piece code {v} at ({r},{c}).")
        return moves

    def legal_moves(self) -> List[Move]:
        """Generate legal moves for the side to move (filters out self-check)."""
        side = self.current_player
        if side not in (1, -1):
            raise InvalidStateError(f"legal_moves: current_player must be +1 or -1, got {side}.")
        self._validate_kings()

        legal: List[Move] = []
        for fr, fc, tr, tc in self.pseudo_legal_moves(side):
            piece = self.board[fr][fc]
            captured = self.board[tr][tc]

            # en passant special during simulation
            ep_captured = None
            ep_restore_cell = None
            if abs(piece) == 1 and (tr, tc) == self.en_passant_square and captured == 0 and fc != tc:
                ep_r, ep_c = fr, tc
                ep_captured = self.board[ep_r][ep_c]
                if abs(ep_captured) != 1 or ep_captured * piece >= 0:
                    continue
                self.board[ep_r][ep_c] = 0
                ep_restore_cell = (ep_r, ep_c)

            # make move
            self.board[tr][tc] = piece
            self.board[fr][fc] = 0

            # castling rook hop for test
            rook_moved = None
            if abs(piece) == 6 and fr == tr and abs(tc - fc) == 2:
                if tc == fc + 2:
                    rr_from, rr_to = (fr, 7), (fr, 5)
                else:
                    rr_from, rr_to = (fr, 0), (fr, 3)
                rook_piece = self.board[rr_from[0]][rr_from[1]]
                self.board[rr_to[0]][rr_to[1]] = rook_piece
                self.board[rr_from[0]][rr_from[1]] = 0
                rook_moved = (rr_from, rr_to, rook_piece)

            ok = not self.in_check(side)

            # undo
            if rook_moved is not None:
                (rr_from, rr_to, rook_piece) = rook_moved
                self.board[rr_from[0]][rr_from[1]] = rook_piece
                self.board[rr_to[0]][rr_to[1]] = 0

            self.board[fr][fc] = piece
            self.board[tr][tc] = captured
            if ep_captured is not None:
                r0, c0 = ep_restore_cell  # type: ignore
                self.board[r0][c0] = ep_captured

            if ok:
                if not (self.in_bounds(fr, fc) and self.in_bounds(tr, tc)):
                    raise OutOfBoundsError(f"Generated legal move has OOB coordinates: {(fr, fc, tr, tc)}")
                legal.append((fr, fc, tr, tc))

        return legal

    # -------------------- NEW: fast legal generator (push/pop) --------------------
    def legal_moves_fast(self) -> List[Move]:
        """Fast legal move generation using push/pop (no manual board fiddling)."""
        side = self.current_player
        if side not in (1, -1):
            raise InvalidStateError(f"legal_moves_fast: current_player must be +1 or -1, got {side}.")
        self._validate_kings()

        out: List[Move] = []
        for mv in self.pseudo_legal_moves(side):
            self.push(mv)
            mover = -self.current_player  # side that just moved
            ok = not self.in_check(mover)
            self.pop()
            if ok:
                out.append(mv)
        return out
    
    # ===================== FACTORED POLICY: PUBLIC API =========================
    def legal_moves_factored(self) -> List[Tuple[int, int]]:
        """
        Return legal moves as (from_idx, delta_idx_0_72) in a white-centric delta space.
        Black's moves are mapped by 180° rotation so the vocabulary is shared.
        """
        return [self.move_to_factored(mv) for mv in self.legal_moves_fast()]

    def masks_factored(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build masks for the current side to move:
        - from_mask: [64]  (1 if any legal delta from that 'from' square)
        - delta_mask: [64, 73]  (1 for each legal delta index from that 'from')
        """
        from_mask = np.zeros(64, dtype=np.bool_)
        delta_mask = np.zeros((64, D73), dtype=np.bool_)
        for mv in self.legal_moves_fast():
            i, d = self.move_to_factored(mv)
            from_mask[i] = True
            if 0 <= d < D73:
                delta_mask[i, d] = True
        return from_mask, delta_mask

    def move_to_factored(self, mv: Move) -> Tuple[int, int]:
        """
        Map a concrete legal move (fr,fc,tr,tc) to (from_idx, delta_idx) in the 73-delta space.
        - Auto-queen promotions map to their geometric SLIDE plane (dr,dc) with step=1,
          which matches AlphaZero-style heads (queen promotion not separated).
        - Castling maps to SLIDE (0,±2).
        """
        fr, fc, tr, tc = mv
        # Canonicalize to white perspective
        if self.current_player == 1:
            fr_w, fc_w, tr_w, tc_w = fr, fc, tr, tc
            from_idx = fr_w * 8 + fc_w
        else:
            fr_w, fc_w = _rot180_rc(fr, fc)
            tr_w, tc_w = _rot180_rc(tr, tc)
            from_idx = fr_w * 8 + fc_w

        dr, dc = tr_w - fr_w, tc_w - fc_w

        # Try SLIDE/KNIGHT lookup first (covers king steps, castles, queen promo geometry)
        key = (dr, dc)
        idx = DELTA_TO_INDEX_73.get(key)
        if idx is not None:
            return from_idx, idx

        # If we ever support underpromotions in the env, we would detect here and map to
        # the proper UNDER_* indices. For now, env auto-queens -> no underpromotion mapping.
        raise InvalidStateError(
            f"move_to_factored: could not map move {mv} with white-perspective delta ({dr},{dc})."
        )

    def factored_to_move(self, from_idx: int, delta_idx: int) -> Move:
        """
        Map (from_idx, delta_idx) back to a board move in the *current* position.
        This does not validate legality; intended for composing priors over legal (i,d) only.
        """
        if not (0 <= from_idx < 64):
            raise OutOfBoundsError(f"from_idx out of range: {from_idx}")

        fr_w, fc_w = divmod(from_idx, 8)
        if not (0 <= delta_idx < D73):
            raise OutOfBoundsError(f"delta_idx out of range: {delta_idx}")

        dr, dc, kind, meta = DELTA_TABLE_73[delta_idx]

        tr_w, tc_w = fr_w + dr, fc_w + dc
        if not self.in_bounds(tr_w, tc_w):
            raise OutOfBoundsError(f"Delta leads OOB: from ({fr_w},{fc_w}) via ({dr},{dc}).")

        if self.current_player == 1:
            fr, fc, tr, tc = fr_w, fc_w, tr_w, tc_w
        else:
            fr, fc = _rot180_rc(fr_w, fc_w)
            tr, tc = _rot180_rc(tr_w, tc_w)

        return (fr, fc, tr, tc)

    # -------------------------------------------------------------------------
    # Piece move generators (pseudo-legal)
    # -------------------------------------------------------------------------
    def _pawn_moves(self, r: int, c: int, piece: int) -> List[Move]:
        """Pawn pushes, captures, and en passant candidates (promotion auto-queen)."""
        if abs(piece) != 1:
            raise InvalidStateError(f"_pawn_moves called with non-pawn {piece} at ({r},{c}).")
        if not self.in_bounds(r, c):
            raise OutOfBoundsError(f"Pawn position out of bounds: ({r},{c}).")

        moves: List[Move] = []
        direction = 1 if piece > 0 else -1
        start_row = 1 if piece > 0 else 6

        # one step forward
        nr = r + direction
        if self.in_bounds(nr, c) and self.board[nr][c] == 0:
            moves.append((r, c, nr, c))
            # two steps forward from start (both squares must be empty)
            if r == start_row:
                nr2 = r + 2 * direction
                if self.in_bounds(nr2, c) and self.board[nr2][c] == 0:
                    moves.append((r, c, nr2, c))

        # diagonal captures and en passant
        for dc in (-1, 1):
            nc = c + dc
            nr = r + direction
            if not self.in_bounds(nr, nc):
                continue
            # normal capture
            if self.board[nr][nc] * piece < 0:
                moves.append((r, c, nr, nc))
            # en passant target (capture onto the passed-over square)
            elif self.en_passant_square == (nr, nc):
                adj = self.board[r][nc]
                if abs(adj) == 1 and adj * piece < 0:
                    moves.append((r, c, nr, nc))
        return moves

    def _knight_moves(self, r: int, c: int, piece: int) -> List[Move]:
        """Knight leaps to 8 L-shaped targets; can capture opposing pieces."""
        if abs(piece) != 2:
            raise InvalidStateError(f"_knight_moves called with non-knight {piece} at ({r},{c}).")
        moves: List[Move] = []
        for dr, dc in [(1, 2), (1, -2), (2, 1), (2, -1),
                       (-1, 2), (-1, -2), (-2, 1), (-2, -1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.board[nr][nc] * piece <= 0:
                moves.append((r, c, nr, nc))
        return moves

    def _bishop_moves(self, r: int, c: int, piece: int) -> List[Move]:
        """Bishop slides along diagonals until blocked."""
        if abs(piece) != 3:
            raise InvalidStateError(f"_bishop_moves called with non-bishop {piece} at ({r},{c}).")
        return self._ray_moves(r, c, piece, [(1, 1), (1, -1), (-1, 1), (-1, -1)])

    def _rook_moves(self, r: int, c: int, piece: int) -> List[Move]:
        """Rook slides along ranks/files until blocked."""
        if abs(piece) != 4:
            raise InvalidStateError(f"_rook_moves called with non-rook {piece} at ({r},{c}).")
        return self._ray_moves(r, c, piece, [(1, 0), (-1, 0), (0, 1), (0, -1)])

    def _queen_moves(self, r: int, c: int, piece: int) -> List[Move]:
        """Queen = bishop + rook rays (call _ray_moves directly to avoid type checks)."""
        if abs(piece) != 5:
            raise InvalidStateError(f"_queen_moves called with non-queen {piece} at ({r},{c}).")
        return self._ray_moves(
            r, c, piece,
            [(1, 1), (1, -1), (-1, 1), (-1, -1),  # bishop-like
             (1, 0), (-1, 0), (0, 1), (0, -1)]    # rook-like
        )

    def _king_moves(self, r: int, c: int, piece: int) -> List[Move]:
        """King single-step moves plus castling (if legal and available)."""
        if abs(piece) != 6:
            raise InvalidStateError(f"_king_moves called with non-king {piece} at ({r},{c}).")

        moves: List[Move] = []
        for dr, dc in [(1, 1), (1, 0), (1, -1), (0, 1),
                       (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.board[nr][nc] * piece <= 0:
                moves.append((r, c, nr, nc))

        # Castling (only from home squares, path empty, and not through/into check)
        opp = -1 if piece > 0 else 1
        if piece > 0 and (r, c) == (0, 4):  # White king on e1
            if self.castling_rights["K"]:
                if self.board[0][5] == 0 and self.board[0][6] == 0:
                    if (not self.is_square_attacked(0, 4, opp) and
                        not self.is_square_attacked(0, 5, opp) and
                        not self.is_square_attacked(0, 6, opp)):
                        if self.board[0][7] == 4:
                            moves.append((0, 4, 0, 6))
            if self.castling_rights["Q"]:
                if self.board[0][3] == 0 and self.board[0][2] == 0 and self.board[0][1] == 0:
                    if (not self.is_square_attacked(0, 4, opp) and
                        not self.is_square_attacked(0, 3, opp) and
                        not self.is_square_attacked(0, 2, opp)):
                        if self.board[0][0] == 4:
                            moves.append((0, 4, 0, 2))

        if piece < 0 and (r, c) == (7, 4):  # Black king on e8
            if self.castling_rights["k"]:
                if self.board[7][5] == 0 and self.board[7][6] == 0:
                    if (not self.is_square_attacked(7, 4, opp) and
                        not self.is_square_attacked(7, 5, opp) and
                        not self.is_square_attacked(7, 6, opp)):
                        if self.board[7][7] == -4:
                            moves.append((7, 4, 7, 6))
            if self.castling_rights["q"]:
                if self.board[7][3] == 0 and self.board[7][2] == 0 and self.board[7][1] == 0:
                    if (not self.is_square_attacked(7, 4, opp) and
                        not self.is_square_attacked(7, 3, opp) and
                        not self.is_square_attacked(7, 2, opp)):
                        if self.board[7][0] == -4:
                            moves.append((7, 4, 7, 2))

        return moves

    def _ray_moves(self, r: int, c: int, piece: int, dirs: List[Tuple[int, int]]) -> List[Move]:
        """Generic sliding move generator for bishops/rooks/queens."""
        if not self.in_bounds(r, c):
            raise OutOfBoundsError(f"Ray piece origin out of bounds: ({r},{c}).")
        if piece == 0:
            raise InvalidStateError(f"_ray_moves called with empty square at ({r},{c}).")

        moves: List[Move] = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            while self.in_bounds(nr, nc):
                target = self.board[nr][nc]
                if target == 0:
                    moves.append((r, c, nr, nc))
                elif target * piece < 0:
                    moves.append((r, c, nr, nc))
                    break
                else:
                    break
                nr += dr
                nc += dc
        return moves

    # -------------------------------------------------------------------------
    # Attack map (do NOT call *_moves here)
    # -------------------------------------------------------------------------
    def _pawn_attacks(self, r: int, c: int, side: int) -> List[Tuple[int, int]]:
        """Squares attacked by a pawn of `side` located at (r, c)."""
        if side not in (1, -1):
            raise InvalidStateError(f"_pawn_attacks side must be +1 or -1, got {side}.")
        out: List[Tuple[int, int]] = []
        dr = 1 if side > 0 else -1
        for dc in (-1, 1):
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                out.append((nr, nc))
        return out

    def _knight_attacks(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Squares attacked by a knight located at (r, c)."""
        out: List[Tuple[int, int]] = []
        for dr, dc in [(1, 2), (1, -2), (2, 1), (2, -1),
                       (-1, 2), (-1, -2), (-2, 1), (-2, -1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                out.append((nr, nc))
        return out

    def _king_attacks(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Squares attacked by a king located at (r, c)."""
        out: List[Tuple[int, int]] = []
        for dr, dc in [(1, 1), (1, 0), (1, -1), (0, 1),
                       (0, -1), (-1, 1), (-1, 0), (-1, -1)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                out.append((nr, nc))
        return out

    def _ray_attacks(self, r: int, c: int, dirs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Generic sliding attack map (includes first opposing blocker square)."""
        out: List[Tuple[int, int]] = []
        me = self.board[r][c]
        if me == 0:
            return out

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            while self.in_bounds(nr, nc):
                target = self.board[nr][nc]
                if target == 0:
                    out.append((nr, nc))
                else:
                    if target * me < 0:
                        out.append((nr, nc))
                    break
                nr += dr
                nc += dc
        return out

    def attacks_of(self, side: int) -> Set[Tuple[int, int]]:
        """Compute the set of squares attacked by `side` (ignores pins)."""
        if side not in (1, -1):
            raise InvalidStateError(f"attacks_of: side must be +1 or -1, got {side}.")
        attacked: Set[Tuple[int, int]] = set()
        for r in range(8):
            for c in range(8):
                v = self.board[r][c]
                if v * side <= 0:
                    continue
                ap = abs(v)
                if ap == 1:
                    attacked.update(self._pawn_attacks(r, c, side))
                elif ap == 2:
                    attacked.update(self._knight_attacks(r, c))
                elif ap == 3:
                    attacked.update(self._ray_attacks(r, c, [(1, 1), (1, -1), (-1, 1), (-1, -1)]))
                elif ap == 4:
                    attacked.update(self._ray_attacks(r, c, [(1, 0), (-1, 0), (0, 1), (0, -1)]))
                elif ap == 5:
                    attacked.update(self._ray_attacks(
                        r, c,
                        [(1, 1), (1, -1), (-1, 1), (-1, -1),
                         (1, 0), (-1, 0), (0, 1), (0, -1)]
                    ))
                elif ap == 6:
                    attacked.update(self._king_attacks(r, c))
                else:
                    raise InvalidStateError(f"Unknown piece code {v} at ({r},{c}).")
        return attacked

    def is_square_attacked(self, r: int, c: int, attacker: int) -> bool:
        """Return True if square (r, c) is attacked by side `attacker`."""
        if not self.in_bounds(r, c):
            raise OutOfBoundsError(f"is_square_attacked queried with OOB square ({r},{c}).")
        if attacker not in (1, -1):
            raise InvalidStateError(f"is_square_attacked attacker must be +1 or -1, got {attacker}.")
        return (r, c) in self.attacks_of(attacker)

    # -------------------------------------------------------------------------
    # Draw / material logic
    # -------------------------------------------------------------------------
    def _insufficient_material(self) -> bool:
        """Return True iff checkmate is impossible with perfect play."""
        pawns = rooks = queens = 0
        w_b = w_n = b_b = b_n = 0

        for r in range(8):
            for c in range(8):
                v = self.board[r][c]
                if v == 0:
                    continue
                av = abs(v)
                if av == 1: pawns += 1
                elif av == 4: rooks += 1
                elif av == 5: queens += 1
                elif av == 2:
                    if v > 0: w_n += 1
                    else: b_n += 1
                elif av == 3:
                    if v > 0: w_b += 1
                    else: b_b += 1

        # Any pawn/rook/queen on board means mating material exists somewhere.
        if pawns or rooks or queens:
            return False

        # Only minors remain
        total_minors = w_b + w_n + b_b + b_n

        # K vs K; K vs K+N; K vs K+B
        if total_minors <= 1:
            return True

        # Exactly two minors total
        if total_minors == 2:
            # Two knights on one side cannot force mate vs bare king
            if (w_n == 2 and w_b == 0 and b_b == 0 and b_n == 0) or \
            (b_n == 2 and b_b == 0 and w_b == 0 and w_n == 0):
                return True
            # One minor each (B vs N, B vs B, N vs N) cannot force mate
            if (w_b + w_n) == 1 and (b_b + b_n) == 1:
                return True
            # Otherwise (BB vs K) or (BN vs K) is *not* insufficient
            return False

        # Three or more minors -> assume sufficient (covers BB, BN, etc.)
        return False


    # -------------------------------------------------------------------------
    # Check / game flow
    # -------------------------------------------------------------------------
    def find_king(self, player: int) -> Optional[Tuple[int, int]]:
        """Locate the king of `player` and return its (row, col); None if missing."""
        if player not in (1, -1):
            raise InvalidStateError(f"find_king: player must be +1 or -1, got {player}.")
        count = 0
        pos = None
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == 6 * player:
                    count += 1
                    pos = (r, c)
                    if count > 1:
                        raise MultipleKingsError(f"Multiple kings for player {player} detected.")
        return pos  # may be None

    def in_check(self, player: int) -> bool:
        """Return True if `player`'s king is under attack."""
        kp = self.find_king(player)
        if kp is None:
            return True
        kr, kc = kp
        return self.is_square_attacked(kr, kc, -player)

    def _preflight_validate_action(self, action: Move) -> None:
        if not (isinstance(action, tuple) or isinstance(action, list)):
            raise InvalidMoveFormatError(f"Action must be a tuple/list of length 4, got {type(action)}.")
        if len(action) != 4:
            raise InvalidMoveFormatError(f"Action must have 4 integers (fr,fc,tr,tc), got length {len(action)}.")
        fr, fc, tr, tc = action
        if not all(isinstance(x, int) for x in (fr, fc, tr, tc)):
            raise InvalidMoveFormatError("All move coordinates must be integers.")
        if not (self.in_bounds(fr, fc) and self.in_bounds(tr, tc)):
            raise OutOfBoundsError(f"Move coordinates out of bounds: {action}.")
        piece = self.board[fr][fc]
        if piece == 0:
            raise EmptySquareError(f"No piece to move at source square ({fr},{fc}).")
        if (piece > 0 and self.current_player != 1) or (piece < 0 and self.current_player != -1):
            raise NotYourTurnError(f"Piece at ({fr},{fc}) belongs to the other side.")
        if self.board[tr][tc] * piece > 0:
            raise WrongPieceColorError(f"Destination ({tr},{tc}) occupied by own piece.")

    def step(self, action: Move) -> Tuple[Board, int, bool, Dict[str, str]]:
        """Apply a *legal* move and advance the game one ply (also updates TT keys)."""
        if self.done:
            raise GameAlreadyFinishedError("Game already finished")
        self._validate_full_state(strict=False)
        self._preflight_validate_action(action)

        legal = set(self.legal_moves())
        if action not in legal:
            raise IllegalMoveError(f"Illegal move: {action}")

        fr, fc, tr, tc = action
        piece = self.board[fr][fc]
        target_before = self.board[tr][tc]

        is_capture = target_before != 0
        is_pawn_move = (abs(piece) == 1)

        # ZOBRIST: remove old EP before any changes
        self._xor_ep_if_any(self.current_player, add=False)

        # en passant capture
        if abs(piece) == 1 and (tr, tc) == self.en_passant_square and self.board[tr][tc] == 0 and fc != tc:
            if abs(self.board[fr][tc]) != 1:
                raise InvalidStateError("En passant target set but no capturable pawn on adjacent file.")
            # ZOBRIST: remove the captured pawn at (fr,tc)
            self._xor_piece(self.board[fr][tc], fr, tc)
            self.board[fr][tc] = 0
            is_capture = True

        # ZOBRIST: move piece from (fr,fc) to (tr,tc) and handle capture
        self._xor_piece(piece, fr, fc)
        if target_before != 0:
            self._xor_piece(target_before, tr, tc)
        self.board[tr][tc] = piece
        self.board[fr][fc] = 0
        self._xor_piece(piece, tr, tc)

        # castling rook hop
        if abs(piece) == 6 and fr == tr and abs(tc - fc) == 2:
            if tc == fc + 2:
                rr_from, rr_to = (fr, 7), (fr, 5)
            else:
                rr_from, rr_to = (fr, 0), (fr, 3)
            rook = self.board[rr_from[0]][rr_from[1]]
            if (piece > 0 and rook != 4) or (piece < 0 and rook != -4):
                raise InvalidStateError("Castling attempted but rook not on expected square.")
            # ZOBRIST rook move
            self._xor_piece(rook, rr_from[0], rr_from[1])
            self._xor_piece(rook, rr_to[0], rr_to[1])
            self.board[rr_to[0]][rr_to[1]] = rook
            self.board[rr_from[0]][rr_from[1]] = 0

        # promotion
        if abs(piece) == 1:
            if piece > 0 and tr == 7:
                # replace pawn with queen in Zobrist
                self._xor_piece(piece, tr, tc)
                self.board[tr][tc] = 5
                self._xor_piece(5, tr, tc)
            elif piece < 0 and tr == 0:
                self._xor_piece(piece, tr, tc)
                self.board[tr][tc] = -5
                self._xor_piece(-5, tr, tc)

        # en passant square update
        if abs(piece) == 1 and abs(tr - fr) == 2:
            self.en_passant_square = ((fr + tr) // 2, fc)
            if self.en_passant_square[0] not in (2, 5):
                raise InvalidStateError(f"Computed en_passant_square invalid: {self.en_passant_square}.")
        else:
            self.en_passant_square = None

        # castling rights updates (+ Zobrist diff)
        before_cr = dict(self.castling_rights)
        if piece == 6:
            self.castling_rights["K"] = False
            self.castling_rights["Q"] = False
        elif piece == -6:
            self.castling_rights["k"] = False
            self.castling_rights["q"] = False

        if piece == 4:
            if (fr, fc) == (0, 0):
                self.castling_rights["Q"] = False
            elif (fr, fc) == (0, 7):
                self.castling_rights["K"] = False
        elif piece == -4:
            if (fr, fc) == (7, 0):
                self.castling_rights["q"] = False
            elif (fr, fc) == (7, 7):
                self.castling_rights["k"] = False

        if target_before == -4:
            if (tr, tc) == (7, 0):
                self.castling_rights["q"] = False
            elif (tr, tc) == (7, 7):
                self.castling_rights["k"] = False
        elif target_before == 4:
            if (tr, tc) == (0, 0):
                self.castling_rights["Q"] = False
            elif (tr, tc) == (0, 7):
                self.castling_rights["K"] = False

        self._xor_castling_bits(before_cr, self.castling_rights)

        # halfmove clock
        if is_capture or is_pawn_move:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # ZOBRIST: flip side
        self._xor_side()
        # ZOBRIST: add new EP (canonical relative to new side)
        self._xor_ep_if_any(-self.current_player, add=True)

        # switch turn
        self.current_player *= -1

        rep_count = self._increment_position_repetition()

        self._validate_full_state(strict=False)

        if self.in_check(self.current_player):
            if not self.legal_moves():
                self.done = True
                self.info = {"result": "checkmate"}
                return self.board, 1, True, self.info
        else:
            if not self.legal_moves():
                self.done = True
                self.info = {"result": "stalemate"}
                return self.board, 0, True, self.info

        if self._insufficient_material():
            self.done = True
            self.info = {"result": "insufficient"}
            return self.board, 0, True, self.info

        if self.halfmove_clock >= 100:
            self.done = True
            self.info = {"result": "fifty-move"}
            return self.board, 0, True, self.info

        if rep_count >= 3:
            self.done = True
            self.info = {"result": "threefold"}
            return self.board, 0, True, self.info

        return self.board, 0, False, {}

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------
    def render(self) -> None:
        """Pretty-print the board as a grid of piece codes (for debugging)."""
        self._validate_board_matrix()
        for row in self.board:
            print(" ".join(f"{x:2}" for x in row))
        print()
