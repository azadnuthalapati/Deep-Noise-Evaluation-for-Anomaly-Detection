import torch
import torch.nn as nn


def hidden_width(d):
    # Appendix E: 64 hidden units for low-dim data, 256 for high-dim (d > 64)
    return 64 if d <= 64 else 256


class VanillaMLP(nn.Module):
    # 4 linear layers with ReLU: d -> h -> h -> h -> d
    def __init__(self, d, hidden=None):
        super().__init__()
        h = hidden or hidden_width(d)
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, d),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.fc2(self.act(self.fc1(x))))


class ResMLP(nn.Module):
    # input projection -> 5 residual blocks -> output projection
    def __init__(self, d, hidden=None, n_blocks=5):
        super().__init__()
        h = hidden or hidden_width(d)
        self.inp = nn.Linear(d, h)
        self.blocks = nn.ModuleList(ResBlock(h) for _ in range(n_blocks))
        self.out = nn.Linear(h, d)

    def forward(self, x):
        x = self.inp(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)


def build_model(d, arch="mlp"):
    if arch == "mlp":
        return VanillaMLP(d)
    if arch == "resmlp":
        return ResMLP(d)
    raise ValueError("arch must be 'mlp' or 'resmlp'")
