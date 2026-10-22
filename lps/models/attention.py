"""注意力模块。

三个模块都做成了「通道数自适应」：初始化时不需要知道输入通道，
首次前向时按实际张量形状构建参数。原因是 ultralytics 的 parse_model
会对自定义模块走 ``c2 = ch[f]`` 分支，yaml 里写的通道数不会自动
按模型宽度缩放，硬编码必然对不上（yolov8n 的宽度系数是 0.25）。

模块用法（yaml 中直接写空参数即可）::

    - [-1, 1, CBAM, []]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CBAM", "CoordAtt", "ECA", "inject_modules"]


class _LazyChannels(nn.Module):
    """通道数延迟构建的基类。"""

    def __init__(self) -> None:
        super().__init__()
        self._built_channels = 0

    def _ensure_built(self, channels: int) -> None:
        if self._built_channels == channels:
            return
        self._build(channels)
        self._built_channels = channels

    def _build(self, channels: int) -> None:  # pragma: no cover - 由子类实现
        raise NotImplementedError


class ChannelAttention(_LazyChannels):
    """CBAM 的通道注意力：平均池化与最大池化共享一个 MLP。"""

    def __init__(self, reduction: int = 16) -> None:
        super().__init__()
        self.reduction = reduction
        self.mlp: nn.Sequential | None = None

    def _build(self, channels: int) -> None:
        hidden = max(8, channels // self.reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_built(x.shape[1])
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        maxv = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + maxv)


class SpatialAttention(_LazyChannels):
    """CBAM 的空间注意力：沿通道维取均值与最大值，再卷积成单通道权重图。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        k = kernel_size if kernel_size in (3, 7) else 7
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def _build(self, channels: int) -> None:
        return  # 与输入通道无关

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.max(x, dim=1, keepdim=True).values
        weight = torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))
        return x * weight


class CBAM(_LazyChannels):
    """Convolutional Block Attention Module（通道 + 空间串联）。"""

    def __init__(self, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(reduction)
        self.sa = SpatialAttention(kernel_size)

    def _build(self, channels: int) -> None:
        self.ca._ensure_built(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class CoordAtt(_LazyChannels):
    """Coordinate Attention：把二维池化拆成横向与纵向两条，保留位置信息。

    结构按原论文实现：两条池化结果拼接后经共享降维层，再分成两路
    各自升维，最后广播相乘。分解成两路时通道维必须独立，
    否则 H≠W 时形状对不上。
    """

    def __init__(self, reduction: int = 32) -> None:
        super().__init__()
        self.reduction = reduction

    def _build(self, channels: int) -> None:
        hidden = max(8, channels // self.reduction)
        self.conv_reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_built(x.shape[1])
        _, _, h, w = x.shape
        pool_h = F.adaptive_avg_pool2d(x, (h, 1))                 # (B,C,H,1)
        pool_w = F.adaptive_avg_pool2d(x, (1, w)).transpose(2, 3)  # (B,C,W,1)
        y = torch.cat([pool_h, pool_w], dim=2)                    # (B,C,H+W,1)
        y = self.act(self.bn(self.conv_reduce(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))                     # (B,C,H,1)
        a_w = torch.sigmoid(self.conv_w(y_w)).transpose(2, 3)     # (B,C,1,W)
        return x * a_h * a_w


class ECA(_LazyChannels):
    """Efficient Channel Attention：一维卷积代替全连接，参数量近似为零。"""

    def __init__(self, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        self.gamma = gamma
        self.b = b
        self.conv: nn.Conv1d | None = None

    def _build(self, channels: int) -> None:
        t = int(abs((math.log2(channels) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_built(x.shape[1])
        y = F.adaptive_avg_pool2d(x, 1)                    # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(1, 2))       # (B,1,C)
        y = y.transpose(1, 2).unsqueeze(-1)
        return x * torch.sigmoid(y)


def inject_modules() -> None:
    """把自定义模块注册进 ultralytics 的模块命名空间。

    parse_model 通过 ``globals()[m]`` 查找模块名，因此只要把类对象
    塞进 ``ultralytics.nn.tasks`` 的全局字典，yaml 里就能直接引用。
    """
    import ultralytics.nn.tasks as tasks

    for name, cls in (("CBAM", CBAM), ("CoordAtt", CoordAtt), ("ECA", ECA)):
        setattr(tasks, name, cls)

    # 重新导出到 ultralytics.nn 顶层，便于外部 from ultralytics.nn import CBAM
    try:
        import ultralytics.nn as ult_nn

        for name, cls in (("CBAM", CBAM), ("CoordAtt", CoordAtt), ("ECA", ECA)):
            setattr(ult_nn, name, cls)
    except ImportError:  # pragma: no cover
        pass
