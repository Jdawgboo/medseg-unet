from __future__ import annotations
from torch import Tensor, nn

class Block(nn.Module):
    def __init__(self, left: int, right: int):
        super().__init__(); self.layers = nn.Sequential(nn.Conv2d(left, right, 3, padding=1), nn.BatchNorm2d(right), nn.ReLU(), nn.Conv2d(right, right, 3, padding=1), nn.BatchNorm2d(right), nn.ReLU())
    def forward(self, value: Tensor) -> Tensor: return self.layers(value)

class UNet(nn.Module):
    """Compact binary 2D U-Net for images shaped [batch, 1, height, width]."""
    def __init__(self, channels: int = 1, base: int = 8):
        super().__init__()
        self.enc1, self.pool1 = Block(channels, base), nn.MaxPool2d(2)
        self.enc2, self.pool2 = Block(base, base * 2), nn.MaxPool2d(2)
        self.mid = Block(base * 2, base * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), Block(base * 4, base * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(base * 2, base, 2, 2), Block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)
    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 1: raise ValueError("expected [batch, 1, height, width]")
        first = self.enc1(images); second = self.enc2(self.pool1(first)); middle = self.mid(self.pool2(second))
        return self.head(self.dec1(__import__('torch').cat([self.up1(self.dec2(__import__('torch').cat([self.up2(middle), second], 1))), first], 1)))
