import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)

class PolicyValueNetFactored(nn.Module):
    """
    Input:   [B, 18, 8, 8]
    Policy:  from_logits  [B, 64]
             delta_logits [B, 64, 73]   (73 canonical deltas per from-square)
    Value:   [B, 1] (tanh)
    """
    def __init__(self, channels: int = 64, n_blocks: int = 6, n_deltas: int = 73):
        super().__init__()
        self.n_deltas = n_deltas

        # Trunk
        self.stem = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])

        # --- Factored policy heads ---
        # From-square head: 1×1 conv -> 1 channel -> flatten to 64
        self.p_from_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),  # bias OK for logits
        )
        # Delta head: 1×1 conv -> 73 channels; later reshape to [B, 64, 73]
        self.p_delta_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_deltas, 1, bias=True),
        )

        # Value head (unchanged)
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.v_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.v_fc2 = nn.Linear(128, 1)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)

        # Factored policy
        p_from = self.p_from_head(x)             # [B, 1, 8, 8]
        p_from = p_from.view(x.size(0), 64)      # [B, 64]

        p_delta = self.p_delta_head(x)           # [B, 73, 8, 8]
        # Rearrange to [B, 64, 73]: one 73-vector per from-square
        p_delta = p_delta.view(x.size(0), self.n_deltas, 64) \
                         .transpose(1, 2).contiguous()      # [B, 64, 73]

        # Value
        v = self.v_head(x).reshape(x.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))            # [B, 1]

        # Return logits (no softmax here; masks/softmax are applied in the agent/adapter)
        return p_from, p_delta, v
# Back-compat alias so other code can import PolicyValueNet
PolicyValueNet = PolicyValueNetFactored