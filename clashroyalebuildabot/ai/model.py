"""The policy network. Needs PyTorch, so only training imports this."""

import torch
from torch import nn

from clashroyalebuildabot.ai.features import GRID_CHANNELS, N_CHOICES, VEC_SIZE
from clashroyalebuildabot.constants import DECK_SIZE


def _conv(cin, cout, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class PolicyNet(nn.Module):
    """Reads the arena grid and the vec; outputs

    card_logits (B, 9): which deck card to play, or wait (last)
    tile_logits (B, 8, 32, 18): where to put each deck card
    """

    def __init__(self, width: int = 64):
        super().__init__()
        self.grid_net = nn.Sequential(
            _conv(GRID_CHANNELS, width),
            _conv(width, width, dilation=2),
            _conv(width, width, dilation=4),
            _conv(width, width, dilation=8),
        )
        # A summary of the whole arena, given to every tile, so placement
        # can react to units far away (e.g. a hog at the other end).
        self.global_net = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.ReLU(inplace=True),
        )
        self.vec_net = nn.Sequential(
            nn.Linear(VEC_SIZE, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, width),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            _conv(3 * width, width),
            _conv(width, width, dilation=2),
        )
        self.tile_head = nn.Conv2d(width, DECK_SIZE, 1)
        self.card_head = nn.Sequential(
            nn.Linear(3 * width, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, N_CHOICES),
        )

    def forward(self, grid, vec):
        g = self.grid_net(grid)
        v = self.vec_net(vec)
        summary = self.global_net(
            torch.cat([g.mean(dim=(2, 3)), g.amax(dim=(2, 3))], dim=1)
        )
        h, w = g.shape[2], g.shape[3]
        v_map = v[:, :, None, None].expand(-1, -1, h, w)
        s_map = summary[:, :, None, None].expand(-1, -1, h, w)
        f = self.fuse(torch.cat([g, v_map, s_map], dim=1))
        tile_logits = self.tile_head(f)
        pooled = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3)), v], dim=1)
        card_logits = self.card_head(pooled)
        return card_logits, tile_logits
