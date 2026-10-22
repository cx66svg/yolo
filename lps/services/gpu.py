"""GPU 资源查询。

训练吃满显卡时，用户最想知道的是「显存还剩多少、利用率多少、会不会 OOM」。
torch 只能拿到显存，拿不到利用率与温度，所以直接问 nvidia-smi。

单次调用约 170ms，绝不能在界面线程里做——见 ui/gpu.py 的线程封装。
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

# 无控制台窗口标志：Windows 下不弹黑框
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
    "--format=csv,noheader,nounits",
]


@dataclass(frozen=True)
class GpuStats:
    index: int
    name: str
    utilization: int      # %
    mem_used: int         # MiB
    mem_total: int        # MiB
    temperature: int      # 摄氏度

    @property
    def mem_percent(self) -> float:
        return self.mem_used / self.mem_total * 100 if self.mem_total else 0.0

    def summary(self) -> str:
        return (f"GPU{self.index} {self.utilization}% · "
                f"显存 {self.mem_used}/{self.mem_total} MB · {self.temperature}°C")


def query_gpus() -> List[GpuStats]:
    """读取所有显卡的实时状态；没有显卡或驱动异常时返回空列表。"""
    try:
        result = subprocess.run(
            _QUERY, capture_output=True, text=True, timeout=4,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    gpus: List[GpuStats] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(GpuStats(
                index=int(parts[0]),
                name=parts[1],
                utilization=int(float(parts[2])),
                mem_used=int(float(parts[3])),
                mem_total=int(float(parts[4])),
                temperature=int(float(parts[5])),
            ))
        except ValueError:
            continue
    return gpus


def has_cuda() -> bool:
    """torch 是否可用 CUDA（决定界面要不要显示 GPU 监控）。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError, AssertionError):
        return False


def primary_gpu() -> Optional[GpuStats]:
    gpus = query_gpus()
    return gpus[0] if gpus else None


def free_memory_mb(index: int = 0) -> float:
    """返回指定显卡的空闲显存（MB）；查不到时返回 -1。

    训练前用它判断显存够不够。0 号卡上常有浏览器等程序占着，
    用户看到「8GB 显卡」就以为够用，结果第一次参数更新就 OOM。
    """
    gpus = query_gpus()
    if not gpus:
        return -1.0
    for gpu in gpus:
        if gpu.index == index:
            return float(gpu.mem_total - gpu.mem_used)
    return float(gpus[0].mem_total - gpus[0].mem_used)
