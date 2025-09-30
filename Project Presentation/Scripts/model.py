import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A standard 2-layer residual block operating on (B, C, 8, 8) tensors.

    Structure:
        x ── Conv3x3 ─ BN ─ ReLU ─ Conv3x3 ─ BN ─(+x)─ ReLU ─→ out

    Notes:
        - Padding=1 keeps spatial size (8x8) constant.
        - No projection on the skip path since in/out channels match.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv-bn-relu
        y = F.relu(self.b1(self.c1(x)))
        # Second conv-bn, then add skip and activate
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNetFactored(nn.Module):
    """
    Policy-Value network with a factored policy head for board games on 8x8 grids.

    Input:
        x: Float tensor of shape [B, 18, 8, 8]
           (18 input feature planes; adjust the stem if you change this.)

    Outputs:
        p_from:  Float logits over from-squares, shape [B, 64]
        p_delta: Float logits of deltas conditioned on from-square, shape [B, 64, n_deltas]
        v:       Float value in [-1, 1], shape [B, 1] (via tanh)

    Usage:
        - Do NOT apply softmax inside the model. Downstream code should:
          * mask illegal from-squares and deltas,
          * apply softmax(s) where appropriate.
        - The joint policy over moves can be formed as:
          softmax(p_from)[i] * softmax(p_delta[i, :])[delta],
          after masking.
    """
    def __init__(self, channels: int = 64, n_blocks: int = 6, n_deltas: int = 73):
        """
        Args:
            channels:  width of the residual trunk.
            n_blocks:  number of residual blocks in the trunk.
            n_deltas:  number of canonical deltas per from-square (e.g., 73 for chess).
        """
        super().__init__()
        self.n_deltas = n_deltas

        # ----- Shared convolutional trunk -----
        # Stem: lift 18 planes to `channels`, keep 8x8 spatial resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Residual stack
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

        # ----- Factored policy heads -----
        # From-square head: produce a single 8x8 map (→ 64 logits).
        self.p_from_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),  # bias allowed for logits
        )
        # Delta head: produce n_deltas channels per 8x8 location (→ 64 × n_deltas).
        self.p_delta_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_deltas, kernel_size=1, bias=True),
        )

        # ----- Value head -----
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.v_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.v_fc2 = nn.Linear(128, 1)

        # ----- Initialization -----
        # He init for convs; He-uniform for linears (a=0.01 corresponds to leaky slope).
        # Biases are zeroed.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: [B, 18, 8, 8] float tensor

        Returns:
            p_from:  [B, 64]   (from-square logits)
            p_delta: [B, 64, n_deltas]  (delta logits per from-square)
            v:       [B, 1]    (value in [-1, 1])
        """
        # Shared trunk
        x = self.stem(x)
        x = self.trunk(x)

        # ----- Factored policy -----
        # From-square logits: [B, 1, 8, 8] → [B, 64]
        p_from = self.p_from_head(x).view(x.size(0), 64)

        # Delta logits: [B, n_deltas, 8, 8] → [B, 64, n_deltas]
        p_delta = self.p_delta_head(x)                                   # [B, n_deltas, 8, 8]
        p_delta = p_delta.view(x.size(0), self.n_deltas, 64)             # [B, n_deltas, 64]
        p_delta = p_delta.transpose(1, 2).contiguous()                   # [B, 64, n_deltas]

        # ----- Value head -----
        v = self.v_head(x).reshape(x.size(0), -1)                        # [B, 16*8*8]
        v = F.relu(self.v_fc1(v))                                        # [B, 128]
        v = torch.tanh(self.v_fc2(v))                                    # [B, 1], in [-1, 1]

        # Return raw logits for policy; downstream applies masks/softmax.
        return p_from, p_delta, v


# Back-compat alias so other code can import PolicyValueNet
PolicyValueNet = PolicyValueNetFactored
