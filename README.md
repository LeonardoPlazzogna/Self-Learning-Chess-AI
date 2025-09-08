# selfplay-chess-ai

AlphaZero-style chess engine that learns from self-play. MCTS guided by a PyTorch policy/value network; exports PGNs and (X, Π, Z) for training/eval.

## Status
WIP — code incoming; repo scaffolded.

## Requirements
- Python 3.11+
- PyTorch (install per your CUDA/CPU setup)

## Quick start
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt

# --- create folders ---
New-Item -ItemType Directory -Force -Path src\chess_ai\cli | Out-Null
New-Item -ItemType Directory -Force -Path models, tests, docs | Out-Null
New-Item -ItemType File -Force -Path models\.gitkeep | Out-Null

# --- library files ---
@'
"""Chess AI package root."""
__all__ = ["env", "mcts", "net", "train_loop"]
'@ | Set-Content -Encoding UTF8 src\chess_ai\__init__.py

@'
class ChessEnv:
    """Minimal scaffold for your chess environment."""

    def reset(self):
        """Return initial state."""
        raise NotImplementedError("Implement board setup")

    def step(self, action):
        """Apply action; return (next_state, reward, done, info)."""
        raise NotImplementedError("Implement move application")
'@ | Set-Content -Encoding UTF8 src\chess_ai\env.py

@'
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Node:
    state: Any
    parent: Optional["Node"] = None
    action_from_parent: Optional[int] = None
    N: int = 0  # visits
    W: float = 0.0  # total value
    P: float = 0.0  # prior
    children: Dict[int, "Node"] = None

class MCTS:
    """Skeleton MCTS; fill in select/expand/simulate/backprop."""

    def __init__(self, policy_fn=None, value_fn=None, c_puct: float = 1.0):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.c_puct = c_puct

    def search(self, state, n_simulations: int = 100) -> Dict[int, float]:
        """Return a policy over actions (π); placeholder implementation."""
        # TODO: implement MCTS; return normalized visit counts
        return {}
'@ | Set-Content -Encoding UTF8 src\chess_ai\mcts.py

@'
# Placeholder network module; wire up PyTorch later.
class PolicyValueNet:
    def __init__(self, board_shape=(8, 8, 12), n_actions=4672):
        self.board_shape = board_shape
        self.n_actions = n_actions

    def predict(self, state):
        """Return (policy_logits, value) placeholders."""
        raise NotImplementedError("Plug in a real PyTorch model")
'@ | Set-Content -Encoding UTF8 src\chess_ai\net.py

@'
def train_loop(data_dir: str = "datasets", epochs: int = 1):
    """Minimal training loop placeholder."""
    # TODO: load (X, Π, Z), run optimizer steps, save checkpoints
    print(f"[WIP] training for {epochs} epoch(s) on {data_dir}")
'@ | Set-Content -Encoding UTF8 src\chess_ai\train_loop.py

# --- CLI entry points ---
@'
import argparse

def main():
    ap = argparse.ArgumentParser("Self-play generator")
    ap.add_argument("--games", type=int, default=10, help="Number of self-play games")
    args = ap.parse_args()
    # TODO: wire MCTS + Env + Net
    print(f"[WIP] would play {args.games} self-play games")
'@ | Set-Content -Encoding UTF8 src\chess_ai\cli\selfplay.py

@'
import argparse
from chess_ai.train_loop import train_loop

def main():
    ap = argparse.ArgumentParser("Training")
    ap.add_argument("--data", default="datasets", help="Dataset directory")
    ap.add_argument("--epochs", type=int, default=1, help="Epochs")
    args = ap.parse_args()
    train_loop(args.data, args.epochs)
'@ | Set-Content -Encoding UTF8 src\chess_ai\cli\train.py

@'
import argparse

def main():
    ap = argparse.ArgumentParser("Arena matches")
    ap.add_argument("--games", type=int, default=10, help="Number of games")
    args = ap.parse_args()
    # TODO: load two checkpoints and play head-to-head
    print(f"[WIP] would run an arena of {args.games} games")
'@ | Set-Content -Encoding UTF8 src\chess_ai\cli\arena.py

# --- packaging (entry points) ---
@'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "selfplay-chess-ai"
version = "0.0.1"
description = "AlphaZero-style chess engine (WIP)"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

[project.scripts]
chess-selfplay = "chess_ai.cli.selfplay:main"
chess-train = "chess_ai.cli.train:main"
chess-arena = "chess_ai.cli.arena:main"
'@ | Set-Content -Encoding UTF8 pyproject.toml

Write-Host "`nScaffold created. Next steps:"
Write-Host "1) python -m venv .venv; . .\\.venv\\Scripts\\activate"
Write-Host "2) pip install -e ."
Write-Host "3) chess-selfplay --games 2   # try a CLI"
