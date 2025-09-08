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
