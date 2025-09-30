from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any, Hashable
import math
import random
import copy
import numpy as np

Move = Tuple[int, int, int, int]


@dataclass
class SearchConfig:
    # Core MCTS
    n_simulations: int = 400
    c_puct: float = 1.5
    max_depth: int = 128

    # Progressive widening (children <= pw_c * N^pw_alpha)
    pw_c: Optional[float] = None
    pw_alpha: float = 0.5

    # Limit how many priors to keep at each node (top-k by prior). None = all
    topk_expand: Optional[int] = None

    # Self-play exploration controls
    policy_temperature: float = 1.0
    root_dirichlet_alpha: Optional[float] = None
    root_dirichlet_eps: float = 0.25

    # Rollout controls (unused; we evaluate leaves with value_fn)
    warmup_rollout_plies: int = 0
    rollout_limit: int = 0

    # Optional efficiency knobs
    early_stop_enabled: bool = False
    early_stop_check_every: int = 32
    early_stop_min_visits: int = 128
    early_stop_best_ratio: float = 0.60  # stop if best_visits/total >= this

    dynamic_budget_enabled: bool = False
    dynamic_min_sims: int = 64
    dynamic_sharp_moves_threshold: int = 12
    dynamic_wide_moves_threshold: int = 40
    dynamic_sharp_scale: float = 1.3   # fewer legal moves -> scale up sims
    dynamic_wide_scale: float = 0.7    # very wide branching -> scale down sims
    dynamic_opening_scale: float = 0.8 # extra scale for early plies (< 8)
    repeat_penalty_at_root: float = 0.0  # penalize repetition moves at root


class _Node:
    __slots__ = (
        "parent", "move", "player", "prior", "N", "W", "Q",
        "children", "untried", "expanded", "key"
    )

    def __init__(self,
                 parent: Optional["_Node"],
                 move: Optional[Move],
                 player: int,
                 prior: float,
                 key: Optional[Hashable] = None):
        self.parent: Optional[_Node] = parent
        self.move: Optional[Move] = move
        self.player: int = player
        self.prior: float = float(prior)
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.children: Dict[Move, _Node] = {}
        self.untried: List[Tuple[Move, float]] = []
        self.expanded: bool = False
        self.key: Optional[Hashable] = key


class MCTSAgent:
    """
    AlphaZero-style MCTS with:
      - Re-rooting/tree reuse between moves
      - Dirichlet noise at root (train mode)
      - Temperature sampling in train mode, greedy in eval mode
      - Optional progressive widening and top-k expansion
      - Optional early-stop and dynamic per-move budgets
      - Helper to export root visit distribution (for training)
      - **Push/Pop search path** when env supports env.push()/env.pop() (fast path)
      - **Safety guards**: legality checks + pruning to avoid desync errors
      - **Transposition Table (TT)**: caches priors/values by state key to reduce recompute
      - **Optional batched GPU evals**: if `value_fn` exposes batch hooks, we micro-batch leaf evals
    """

    def __init__(self, policy_fn, value_fn, cfg: SearchConfig, eval_batch_size: Optional[int] = None):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.cfg = cfg

        self._root: Optional[_Node] = None
        self._explore: bool = True
        self._ply_by_env_id: Dict[int, int] = {}

        # Transposition table (TT) storing lightweight information:
        #   key -> {"priors": Dict[Move,float], "value": Optional[float]}
        self._tt: Dict[Hashable, Dict[str, Any]] = {}

        # If the value_fn wrapper provides a preferred batch size (e.g., GPU), use it;
        # otherwise allow caller to pass eval_batch_size; fallback to 1 (no batching).
        self._eval_batch_size: int = 1
        if hasattr(self.value_fn, "preferred_batch"):
            try:
                self._eval_batch_size = max(1, int(getattr(self.value_fn, "preferred_batch")))
            except Exception:
                self._eval_batch_size = 1
        if eval_batch_size is not None:
            self._eval_batch_size = max(1, int(eval_batch_size))

    # -------- public API --------
    def train_mode(self):
        self._explore = True

    def eval_mode(self):
        self._explore = False

    def choose_move(self, env, explore: Optional[bool] = None) -> Move:
        # --- Instant mate-in-1 check at root ---
        try:
            for mv in env.legal_moves():
                env2 = self._clone_env(env)
                _, _, done, info = env2.step(mv)
                if done and info.get("result") == "checkmate":
                    # advance root for tree reuse
                    self._prepare_root(env, getattr(env, "current_player", 1), explore=(explore is True))
                    self._advance_root(mv)
                    return mv
        except Exception:
            pass

        if explore is None:
            explore = self._explore

        root_player = getattr(env, "current_player", 1)
        self._prepare_root(env, root_player, explore)

        sims_budget = self._compute_sim_budget(env)

        # If we can micro-batch and env supports push/pop, use batched selection/eval.
        can_pushpop = hasattr(env, "push") and hasattr(env, "pop")
        can_batch = (self._eval_batch_size > 1) and (can_pushpop) and (
            hasattr(self.value_fn, "value_many") or hasattr(self.value_fn, "value_many_features")
        )
        if can_batch:
            self._run_sims_batched_pushpop(env, self._root, root_player, sims_budget)
        else:
            # Run simulations with optional early-stop (single-eval path)
            for i in range(sims_budget):
                if can_pushpop:
                    self._simulate_once_pushpop(env, self._root, root_player)
                else:
                    self._simulate_once_clone(env, self._root, root_player)

                if self.cfg.early_stop_enabled and ((i + 1) % self.cfg.early_stop_check_every == 0):
                    moves, visits = self._root_moves_and_visits()
                    if visits:
                        total = sum(visits)
                        if total >= self.cfg.early_stop_min_visits:
                            if (max(visits) / max(1, total)) >= self.cfg.early_stop_best_ratio:
                                break

        # Choose final move
        moves, visits = self._root_moves_and_visits()
        if not moves:
            raise RuntimeError("No legal moves at root.")

        ply = self._infer_ply(env)
        tau = self.cfg.policy_temperature if (explore and ply < 40) else 0.0
        move = self._sample_from_visits(moves, visits, tau)

        # Advance root to chosen child (tree reuse)
        self._advance_root(move)

        # Update ply counter
        self._ply_by_env_id[id(env)] = ply + 1
        return move

    def get_root_policy(self, tau: float = 1.0) -> Dict[Move, float]:
        """
        Return normalized visit distribution at the root (π target for training).
        If tau == 0, return 1-hot argmax distribution.
        """
        moves, visits = self._root_moves_and_visits()
        if not moves:
            return {}
        v = np.array(visits, dtype=float)
        if tau <= 0.0:
            pi = np.zeros_like(v)
            pi[int(np.argmax(v))] = 1.0
        else:
            x = np.power(np.maximum(v, 1e-12), 1.0 / max(1e-12, tau))
            pi = x / x.sum()
        return {mv: float(p) for mv, p in zip(moves, pi)}

    # -------- simulation: fast (push/pop) --------
    def _simulate_once_pushpop(self, root_env, root_node: _Node, root_perspective: int):
        env = root_env
        node = root_node
        depth = 0
        path: List[_Node] = [node]
        pushes = 0

        try:
            while True:
                depth += 1
                # Terminal?
                if getattr(env, "done", False):
                    v = self._terminal_value(env, root_perspective)
                    self._backup(path, v, root_perspective)
                    return

                # Expand priors if needed (via TT-aware gate)
                if not node.expanded:
                    priors = self._priors_for_env(env)
                    self._expand_node(node, env, priors)

                # Progressive widening gate
                allowed_children = self._allowed_children(node)

                # Try expanding a new child
                if node.untried and len(node.children) < allowed_children:
                    move, prior = node.untried.pop(0)

                    # SAFETY: check current legality before pushing
                    if not self._is_legal(env, move):
                        # Drop move permanently for this node
                        continue

                    try:
                        env.push(move)  # mutate
                    except Exception:
                        # SAFETY: if push fails, skip this move
                        continue
                    pushes += 1

                    child_key = self._state_key(env)
                    child = _Node(node, move, getattr(env, "current_player", -node.player), prior, key=child_key)
                    node.children[move] = child
                    path.append(child)

                    if getattr(env, "done", False):
                        v_terminal = self._terminal_value(env, root_perspective)
                        self._backup(path, v_terminal, root_perspective)
                        return

                    v = self._value_for_env(env, root_perspective, key_hint=child_key)
                    self._backup(path, v, root_perspective)
                    return

                # Otherwise descend best child
                if not node.children:
                    # No children and no untried: treat as leaf
                    v = self._value_for_env(env, root_perspective, key_hint=node.key)
                    self._backup(path, v, root_perspective)
                    return

                move, child = self._best_child(node, env)

                # SAFETY: ensure move still legal; otherwise prune child and retry this depth
                if not self._is_legal(env, move):
                    node.children.pop(move, None)
                    continue

                try:
                    env.push(move)
                except Exception:
                    # SAFETY: prune on failure and retry
                    node.children.pop(move, None)
                    continue
                pushes += 1

                # Update child's key after push (if not set)
                if child.key is None:
                    child.key = self._state_key(env)

                path.append(child)
                node = child

                if getattr(env, "done", False):
                    v_terminal = self._terminal_value(env, root_perspective)
                    self._backup(path, v_terminal, root_perspective)
                    return

                if depth >= self.cfg.max_depth:
                    v = self._value_for_env(env, root_perspective, key_hint=node.key)
                    self._backup(path, v, root_perspective)
                    return
        finally:
            # unwind all pushes we made in this simulation
            for _ in range(pushes):
                try:
                    root_env.pop()
                except Exception:
                    break  # SAFETY: don't die if pop fails

    # -------- simulation: fallback (clone) --------
    def _simulate_once_clone(self, root_env, root_node: _Node, root_perspective: int):
        env = self._clone_env(root_env)
        node = root_node
        depth = 0
        path: List[_Node] = [node]

        while True:
            depth += 1
            if getattr(env, "done", False):
                v = self._terminal_value(env, root_perspective)
                self._backup(path, v, root_perspective)
                return

            if not node.expanded:
                priors = self._priors_for_env(env)
                self._expand_node(node, env, priors)

            allowed_children = self._allowed_children(node)
            if node.untried and len(node.children) < allowed_children:
                move, prior = node.untried.pop(0)

                # SAFETY: verify legality now (env may differ vs. when priors were computed)
                if not self._is_legal(env, move):
                    continue  # discard bad candidate and try next

                try:
                    _, _, done, _ = env.step(move)
                except Exception:
                    # SAFETY: if step raises (e.g., racey EP/castling), discard move
                    continue

                child_key = self._state_key(env)
                child = _Node(node, move, getattr(env, "current_player", -node.player), prior, key=child_key)
                node.children[move] = child
                path.append(child)
                if done:
                    v_terminal = self._terminal_value(env, root_perspective)
                    self._backup(path, v_terminal, root_perspective)
                    return
                v = self._value_for_env(env, root_perspective, key_hint=child_key)
                self._backup(path, v, root_perspective)
                return

            if not node.children:
                # Leaf with nothing to try (should be rare)
                v = self._value_for_env(env, root_perspective, key_hint=node.key)
                self._backup(path, v, root_perspective)
                return

            move, child = self._best_child(node, env)

            # SAFETY: ensure move is legal at this exact env snapshot
            if not self._is_legal(env, move):
                node.children.pop(move, None)  # prune stale child
                continue

            try:
                _, _, done, _ = env.step(move)
            except Exception:
                # SAFETY: prune child on failure and retry
                node.children.pop(move, None)
                continue

            if child.key is None:
                child.key = self._state_key(env)

            path.append(child)
            node = child

            if done:
                v_terminal = self._terminal_value(env, root_perspective)
                self._backup(path, v_terminal, root_perspective)
                return

            if depth >= self.cfg.max_depth:
                v = self._value_for_env(env, root_perspective, key_hint=node.key)
                self._backup(path, v, root_perspective)
                return

    # -------- batched push/pop sims (micro-batching of leaf evals) --------
    def _run_sims_batched_pushpop(self, root_env, root_node: _Node, root_perspective: int, budget: int):
        """
        Collect up to `eval_batch_size` leaves per wave, evaluate them in batch using
        `value_fn.value_many(...)` or `value_fn.value_many_features(...)` if exposed,
        then backup results. Falls back to single eval if batch hooks are missing.
        """
        B = max(1, int(self._eval_batch_size))

        sims_done = 0
        while sims_done < budget:
            wave = min(B, budget - sims_done)

            # Collect leaves (selection + expansion) without evaluating now
            leaf_paths: List[List[_Node]] = []
            leaf_keys: List[Optional[Hashable]] = []
            features_batch: List[Any] = []   # optional features from wrapper
            env_batch: List[Any] = []        # fallback: env snapshots (last resort)
            todo_idx: List[int] = []         # indices needing NN evaluation
            vals: List[Optional[float]] = []

            # Early stop guard check is done after each wave
            for k in range(wave):
                req = self._select_to_leaf_request(root_env, root_node, root_perspective)
                if req is None:
                    # nothing selected (terminal at root), bail
                    continue

                path, key, feat_or_none, env_or_none, val_or_none = req
                leaf_paths.append(path)
                leaf_keys.append(key)
                vals.append(val_or_none)

                if val_or_none is None:
                    # No TT terminal/value; queue for NN
                    todo_idx.append(k)
                    if feat_or_none is not None:
                        features_batch.append(feat_or_none)
                        env_batch.append(None)
                    else:
                        features_batch.append(None)
                        env_batch.append(env_or_none)

            # Evaluate the NN for the pending ones (if any)
            if todo_idx:
                try:
                    if hasattr(self.value_fn, "value_many_features") and any(f is not None for f in features_batch):
                        feats = [features_batch[i] for i in todo_idx]
                        preds = list(self.value_fn.value_many_features(feats, root_perspective))
                    elif hasattr(self.value_fn, "value_many"):
                        envs = [env_batch[i] for i in todo_idx]
                        preds = list(self.value_fn.value_many(envs, root_perspective))
                    else:
                        # Fallback: single calls
                        preds = []
                        for i in todo_idx:
                            if features_batch[i] is not None and hasattr(self.value_fn, "__call_features__"):
                                # Optional: some wrappers expose a __call_features__(feat, persp)
                                preds.append(float(self.value_fn.__call_features__(features_batch[i], root_perspective)))
                            else:
                                preds.append(float(self.value_fn(env_batch[i], root_perspective)))
                except Exception:
                    # Robust fallback: always be able to do scalar calls
                    preds = []
                    for i in todo_idx:
                        preds.append(float(self.value_fn(env_batch[i], root_perspective)))

                # Map predictions back to the global vals list and TT
                for i_local, i_global in enumerate(todo_idx):
                    v = float(preds[i_local])
                    vals[i_global] = v
                    kkey = leaf_keys[i_global]
                    if kkey is not None:
                        ent = self._tt.setdefault(kkey, {})
                        ent["value"] = float(v)

            # Backup all collected paths
            for path, v in zip(leaf_paths, vals):
                if v is None:
                    continue  # should not happen, but guard
                self._backup(path, float(v), root_perspective)

            sims_done += wave

            # Early stop check (after each wave)
            if self.cfg.early_stop_enabled:
                moves, visits = self._root_moves_and_visits()
                if visits:
                    total = sum(visits)
                    if total >= self.cfg.early_stop_min_visits:
                        if (max(visits) / max(1, total)) >= self.cfg.early_stop_best_ratio:
                            break

    def _select_to_leaf_request(self, root_env, root_node: _Node, root_perspective: int):
        """
        One simulation: descend/expand to a leaf and return a request tuple for evaluation:
          (path, leaf_key, features_or_None, env_snapshot_or_None, value_or_None_if_TT_or_terminal)

        Uses push/pop and restores env before returning.
        """
        env = root_env
        node = root_node
        depth = 0
        path: List[_Node] = [node]
        pushes = 0

        try:
            while True:
                depth += 1
                if getattr(env, "done", False):
                    v = self._terminal_value(env, root_perspective)
                    # Direct backup now since it's terminal
                    self._backup(path, v, root_perspective)
                    return None  # indicates nothing to evaluate

                if not node.expanded:
                    priors = self._priors_for_env(env)
                    self._expand_node(node, env, priors)

                allowed_children = self._allowed_children(node)

                if node.untried and len(node.children) < allowed_children:
                    move, prior = node.untried.pop(0)
                    if not self._is_legal(env, move):
                        continue
                    try:
                        env.push(move)
                    except Exception:
                        continue
                    pushes += 1

                    child_key = self._state_key(env)
                    child = _Node(node, move, getattr(env, "current_player", -node.player), prior, key=child_key)
                    node.children[move] = child
                    path.append(child)

                    if getattr(env, "done", False):
                        v_terminal = self._terminal_value(env, root_perspective)
                        self._backup(path, v_terminal, root_perspective)
                        return None

                    # Try TT value first
                    tt_v = self._tt.get(child_key, {}).get("value") if child_key is not None else None
                    if tt_v is not None:
                        # Backup immediately, no NN call needed
                        self._backup(path, float(tt_v), root_perspective)
                        return None

                    # Need NN eval: prefer features hook; else last-resort snapshot
                    feat = self._maybe_features_from_env(env)
                    snapshot = None
                    if feat is None:
                        snapshot = self._clone_env(env)  # last resort; only if wrapper lacks feature hook
                    return (path, child_key, feat, snapshot, None)

                if not node.children:
                    # Leaf reached without untried children -> evaluate this position
                    kkey = node.key if node.key is not None else self._state_key(env)
                    tt_v = self._tt.get(kkey, {}).get("value") if kkey is not None else None
                    if tt_v is not None:
                        self._backup(path, float(tt_v), root_perspective)
                        return None
                    feat = self._maybe_features_from_env(env)
                    snapshot = None
                    if feat is None:
                        snapshot = self._clone_env(env)
                    return (path, kkey, feat, snapshot, None)

                move, child = self._best_child(node, env)
                if not self._is_legal(env, move):
                    node.children.pop(move, None)
                    continue
                try:
                    env.push(move)
                except Exception:
                    node.children.pop(move, None)
                    continue
                pushes += 1

                if child.key is None:
                    child.key = self._state_key(env)

                path.append(child)
                node = child

                if getattr(env, "done", False):
                    v_terminal = self._terminal_value(env, root_perspective)
                    self._backup(path, v_terminal, root_perspective)
                    return None

                if depth >= self.cfg.max_depth:
                    kkey = node.key if node.key is not None else self._state_key(env)
                    tt_v = self._tt.get(kkey, {}).get("value") if kkey is not None else None
                    if tt_v is not None:
                        self._backup(path, float(tt_v), root_perspective)
                        return None
                    feat = self._maybe_features_from_env(env)
                    snapshot = None
                    if feat is None:
                        snapshot = self._clone_env(env)
                    return (path, kkey, feat, snapshot, None)

        finally:
            for _ in range(pushes):
                try:
                    root_env.pop()
                except Exception:
                    break

    # -------- re-rooting / warm start --------
    def _prepare_root(self, env, root_player: int, explore: bool):
        legal = getattr(env, "legal_moves", None)
        legal_moves = set(legal()) if callable(legal) else set()

        can_reuse = (
            self._root is not None
            and self._root.player == root_player
            and all(m in legal_moves for m in self._root.children.keys())
        )

        base_priors = self._priors_for_env(env)
        if explore and (self.cfg.root_dirichlet_alpha is not None):
            priors = self._apply_dirichlet_noise(base_priors, self.cfg.root_dirichlet_alpha, self.cfg.root_dirichlet_eps)
        else:
            priors = base_priors

        root_key = self._state_key(env)
        if not can_reuse:
            self._root = _Node(None, None, root_player, 1.0, key=root_key)
            self._expand_node(self._root, env, priors)
        else:
            self._refresh_root_priors(self._root, env, priors)

    def _advance_root(self, move: Move):
        if self._root is None:
            return
        child = self._root.children.get(move)
        if child is None:
            self._root = None
            return
        child.parent = None
        child.move = None
        self._root = child

    def _refresh_root_priors(self, node: _Node, env, priors: Dict[Move, float]):
        legal = getattr(env, "legal_moves", None)
        legal_moves = list(legal()) if callable(legal) else []
        legal_set = set(legal_moves)

        # If any existing child is now illegal, rebuild
        for mv in list(node.children.keys()):
            if mv not in legal_set:
                node.children.clear()
                node.N = 0
                node.W = 0.0
                node.Q = 0.0
                node.expanded = False
                self._expand_node(node, env, priors)
                return

        # Update priors on existing children
        for mv, ch in node.children.items():
            ch.prior = float(priors.get(mv, 1e-12))

        # Rebuild untried as remaining legal moves
        remaining = [m for m in legal_moves if m not in node.children]
        items = sorted(((m, float(priors.get(m, 1e-12))) for m in remaining),
                       key=lambda kv: kv[1], reverse=True)
        if self.cfg.topk_expand is not None:
            items = items[: self.cfg.topk_expand]
        node.untried = items
        node.expanded = True

    # -------- helpers --------
    def _best_child(self, node: _Node, env) -> Tuple[Move, _Node]:
        sqrt_N = math.sqrt(max(1, node.N))
        c = self.cfg.c_puct
        items = list(node.children.items())
        random.shuffle(items)

        best_score = -1e18
        best = items[0]
        for mv, ch in items:
            u = c * ch.prior * (sqrt_N / (1 + ch.N))
            score = ch.Q + u
            # NEW: repetition penalty at root
            score += self._root_repeat_penalty(env, mv, node)
            if score > best_score:
                best_score = score
                best = (mv, ch)
        return best

    def _expand_node(self, node: _Node, env, priors: Dict[Move, float]):
        node.expanded = True
        items = sorted(priors.items(), key=lambda kv: kv[1], reverse=True)
        if self.cfg.topk_expand is not None:
            items = items[: self.cfg.topk_expand]
        node.untried = items

    def _allowed_children(self, node: _Node) -> int:
        if self.cfg.pw_c is None:
            return 10**9
        return max(1, int(self.cfg.pw_c * (node.N ** self.cfg.pw_alpha)))

    def _backup(self, path: List[_Node], v_leaf_from_root: float, root_perspective: int):
        for node in reversed(path):
            node.N += 1
            node.W += v_leaf_from_root if (node.player == root_perspective) else -v_leaf_from_root
            node.Q = node.W / node.N

    def _clone_env(self, env):
        # Used only when strictly necessary (fallback and batched snapshots without feature hook)
        try:
            return env.clone()
        except AttributeError:
            return copy.deepcopy(env)

    def _terminal_value(self, env, root_perspective: int) -> float:
        info = getattr(env, "info", {}) if hasattr(env, "info") else {}
        res = info.get("result")
        if res == "checkmate":
            winner = -getattr(env, "current_player", -1)
            return 1.0 if winner == root_perspective else -1.0
        return 0.0

    def _normalize_priors(self, priors: Dict[Move, float], env) -> Dict[Move, float]:
        legal = getattr(env, "legal_moves", None)
        moves = legal() if callable(legal) else []
        if not moves:
            return {}
        if not priors:
            p = 1.0 / len(moves)
            return {m: p for m in moves}
        filtered = {m: float(priors.get(m, 0.0)) for m in moves}
        s = sum(x for x in filtered.values() if x > 0)
        if s <= 0:
            p = 1.0 / len(moves)
            return {m: p for m in moves}
        return {m: max(1e-12, x) / s for m, x in filtered.items()}
    
    def _softmax_masked(self, x: np.ndarray, mask: np.ndarray, axis: Optional[int]=None) -> np.ndarray:
        """
        Softmax over logits x with a 0/1 mask; returns probabilities only over masked entries,
        with 0 on masked-out positions.
        """
        x = np.asarray(x, dtype=float)
        m = np.asarray(mask, dtype=float)
        x_masked = np.where(m > 0, x, -1e30)               # very negative for masked-out
        x_shift = x_masked - np.max(x_masked, axis=axis, keepdims=True)
        e = np.exp(x_shift)
        e *= m
        s = np.sum(e, axis=axis, keepdims=True)
        return e / np.maximum(s, 1e-12)


    def _apply_dirichlet_noise(self, priors: Dict[Move, float], alpha: float, eps: float) -> Dict[Move, float]:
        moves = list(priors.keys())
        p = np.array([priors[m] for m in moves], dtype=float)
        noise = np.random.dirichlet([alpha] * len(moves))
        mixed = (1.0 - eps) * p + eps * noise
        mixed /= mixed.sum()
        return {m: float(x) for m, x in zip(moves, mixed)}

    def _root_moves_and_visits(self) -> Tuple[List[Move], List[int]]:
        if self._root is None:
            return [], []
        moves = list(self._root.children.keys())
        visits = [self._root.children[m].N for m in moves]
        return moves, visits

    def _infer_ply(self, env) -> int:
        if hasattr(env, "ply_count"):
            return int(getattr(env, "ply_count"))
        if hasattr(env, "history") and isinstance(env.history, (list, tuple)):
            return len(env.history)
        return self._ply_by_env_id.get(id(env), 0)

    @staticmethod
    def _sample_from_visits(moves: List[Move], visits: List[int], tau: float) -> Move:
        v = np.asarray(visits, dtype=float)
        if v.size == 0:
            raise RuntimeError("No visits to sample from at root.")
        if tau <= 0.0:
            m = v.max()
            ties = np.flatnonzero(v == m)
            idx = int(np.random.choice(ties))
            return moves[idx]
        inv_tau = 1.0 / max(1e-12, tau)
        x = np.power(np.maximum(v, 1e-12), inv_tau)
        s = x.sum()
        if not np.isfinite(s) or s <= 0.0:
            idx = int(np.random.randint(len(moves)))
            return moves[idx]
        p = x / s
        idx = int(np.random.choice(len(moves), p=p))
        return moves[idx]
    
    def _root_repeat_penalty(self, env, move, node: _Node) -> float:
        """Negative penalty if taking `move` at the root would enter a position
        already seen twice (i.e., immediate twofold repetition)."""
        if node.parent is not None:
            return 0.0
        if self.cfg.repeat_penalty_at_root <= 0.0:
            return 0.0
        if not self._explore:
            return 0.0  # only during training explore
        pen = 0.0
        try:
            env.push(move)
            # Same key logic the env uses for repetition
            key = getattr(env, "_position_key", None)
            if callable(key):
                k = env._position_key()
                cnt = int(getattr(env, "position_counts", {}).get(k, 0))
                if cnt >= 2:
                    pen -= float(self.cfg.repeat_penalty_at_root)
        except Exception:
            pass
        finally:
            try:
                env.pop()
            except Exception:
                pass
        return pen

    # -------- internal --------
    def _compute_sim_budget(self, env) -> int:
        if not self.cfg.dynamic_budget_enabled:
            return int(self.cfg.n_simulations)
        try:
            legal_n = len(env.legal_moves())
        except Exception:
            legal_n = 30
        ply = self._infer_ply(env)
        scale = 1.0
        if legal_n <= self.cfg.dynamic_sharp_moves_threshold:
            scale *= self.cfg.dynamic_sharp_scale
        elif legal_n >= self.cfg.dynamic_wide_moves_threshold:
            scale *= self.cfg.dynamic_wide_scale
        if ply < 8:
            scale *= self.cfg.dynamic_opening_scale
        budget = int(self.cfg.n_simulations * scale)
        return max(self.cfg.dynamic_min_sims, budget)

    # -------- legality helper --------
    @staticmethod
    def _is_legal(env, move: Move) -> bool:
        try:
            leg = env.legal_moves()
            # Some envs return lists; convert to set for speed
            return move in (set(leg) if not isinstance(leg, set) else leg)
        except Exception:
            return False

    # -------- TT-aware helpers --------
    def _state_key(self, env) -> Optional[Hashable]:
        # Prefer Zobrist integer if present
        try:
            if hasattr(env, "zkey"):
                z = getattr(env, "zkey")
                if isinstance(z, (int, np.integer)):
                    return ("z", int(z))
        except Exception:
            pass

        # Fallbacks (attributes or methods you may have elsewhere)
        for attr in ("zobrist", "tt_key", "hash_key"):
            try:
                if hasattr(env, attr):
                    val = getattr(env, attr)
                    return ("a", int(val) if isinstance(val, (int, np.integer)) else str(val))
            except Exception:
                continue

        try:
            if hasattr(env, "fen") and callable(env.fen):
                return ("fen", env.fen())
        except Exception:
            pass

        # Safe tuple fallback
        try:
            board_t = tuple(tuple(int(x) for x in row) for row in env.board)
            cp = int(getattr(env, "current_player", 1))
            cr = getattr(env, "castling_rights", {})
            cr_t = (bool(cr.get("K", False)), bool(cr.get("Q", False)),
                    bool(cr.get("k", False)), bool(cr.get("q", False)))
            ep = getattr(env, "en_passant_square", None)
            ep_t = (int(ep[0]), int(ep[1])) if (isinstance(ep, tuple) and len(ep) == 2) else None
            hm = int(getattr(env, "halfmove_clock", 0))
            fm = int(getattr(env, "fullmove_number", 1)) if hasattr(env, "fullmove_number") else 1
            return ("tu", board_t, cp, cr_t, ep_t, hm, fm)
        except Exception:
            return None

    def _priors_for_env(self, env) -> Dict[Move, float]:
        key = self._state_key(env)
        if key is not None:
            ent = self._tt.get(key)
            if ent is not None and ent.get("priors"):
                return self._normalize_priors(ent["priors"], env)

        # Path A: factored head present on policy wrapper
        if hasattr(self.policy_fn, "policy_factored"):
            # Env provides masks over (from, delta) in the canonical 73-delta space
            from_mask, delta_mask = env.masks_factored()  # [64], [64,73] uint8/bool

            # Ask policy for logits for both heads (shape [64] and [64,73])
            from_logits, delta_logits = self.policy_fn.policy_factored(env, from_mask, delta_mask)

            # Masked softmax to get probabilities
            p_from  = self._softmax_masked(from_logits,    from_mask,  axis=0)    # [64]
            p_delta = self._softmax_masked(delta_logits,   delta_mask, axis=1)    # [64,73]

            pri: Dict[Move, float] = {}
            for from_idx, delta_idx in env.legal_moves_factored():
                p = float(p_from[from_idx]) * float(p_delta[from_idx, delta_idx])
                if p <= 0.0:
                    continue
                mv = env.factored_to_move(from_idx, delta_idx)
                pri[mv] = p

            pri = self._normalize_priors(pri, env)
            if key is not None:
                slot = self._tt.setdefault(key, {})
                slot["priors"] = dict(pri)
            return pri

        # Path B: legacy flat policy (unchanged)
        base = self.policy_fn(env)  # Dict[Move,float] or logits your adapter already produced
        pri = self._normalize_priors(base, env)
        if key is not None:
            slot = self._tt.setdefault(key, {})
            slot["priors"] = dict(pri)
        return pri

    def _value_for_env(self, env, root_perspective: int, key_hint: Optional[Hashable] = None) -> float:
        key = key_hint if key_hint is not None else self._state_key(env)
        if key is not None:
            ent = self._tt.get(key)
            if ent is not None and ("value" in ent) and (ent["value"] is not None):
                return float(ent["value"])
        v = float(self.value_fn(env, root_perspective))
        if key is not None:
            slot = self._tt.setdefault(key, {})
            slot["value"] = float(v)
        return v

    def _maybe_features_from_env(self, env):
        # Optional hook for batched GPU eval: wrapper may expose feature extraction
        try:
            if hasattr(self.value_fn, "features_from_env"):
                return self.value_fn.features_from_env(env)
        except Exception:
            pass
        return None