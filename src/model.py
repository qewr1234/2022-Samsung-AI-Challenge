"""Small U-Net for single-channel image-to-image regression (SEM -> depth)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Three-level U-Net. Input is padded so any H/W works; output is cropped back.

    Output passes through a sigmoid so predictions live in [0, 1], matching the
    normalised depth targets.
    """

    def __init__(self, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = _block(1, c1)
        self.enc2 = _block(c1, c2)
        self.enc3 = _block(c2, c3)
        self.bottleneck = _block(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _block(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _block(c1 * 2, c1)
        self.head = nn.Conv2d(c1, 1, 1)
        self._mult = 8  # three 2x downsamplings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph = (-h) % self._mult
        pw = (-w) % self._mult
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = torch.sigmoid(self.head(d1))
        return out[..., :h, :w]
