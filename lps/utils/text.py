"""终端文本清理。

ultralytics 的日志带 ANSI 颜色控制符（如 ``\\x1b[34m\\x1b[1m``）。这些序列在
真实终端里会被解释成颜色，但直接塞进 PyQt 的文本框或写进日志文件，
就会显示成一堆 ``[34m[1m`` 之类的乱码。

这里统一剥离 ANSI 转义序列，并顺带处理进度条残留的 ``\\r`` 与 ``\\x1b[K`` 清屏符。
"""
from __future__ import annotations

import re

# 覆盖 CSI（颜色、光标移动、清行）与 OSC（窗口标题）两类转义序列
_ANSI_RE = re.compile(
    r"""
    \x1b\[[0-?]*[ -/]*[@-~]      # CSI 序列
    | \x1b\][^\x07\x1b]*(?:\x07|\x1b\\)   # OSC 序列
    | \x1b[@-Z\\-_]              # 单字符转义
    """,
    re.VERBOSE,
)

# 进度条用 \r 原地刷新，保留会让文本框里出现大量重复行
_CR_RE = re.compile(r"[^\n]*\r(?=[^\n]*)")


def strip_ansi(text: str) -> str:
    """去掉 ANSI 转义序列。"""
    return _ANSI_RE.sub("", text)


def clean_log_line(line: str) -> str:
    """清理单行日志：去 ANSI、去行内 \\r 回退、去首尾空白。

    进度条一行内会反复 \\r 覆盖，取最后一次覆盖后的内容才是用户真正看到的。
    """
    line = strip_ansi(line)
    if "\r" in line:
        parts = [p for p in line.split("\r") if p.strip()]
        line = parts[-1] if parts else ""
    return line.rstrip()


# 与训练无关、只会刷屏的固定噪音行。只匹配明确的固定串，
# 避免误伤真正的错误信息。
_NOISE_PATTERNS = (
    "UserWarning: adaptive_max_pool2d",
    "return Variable._execution_engine.run_backward",
    "New https://pypi.org/project/ultralytics",
    "Update with 'pip install -U ultralytics'",
    "WARNING No module named 'polars'",
)


def is_noise_line(line: str) -> bool:
    """判断是否是应当从界面日志里过滤掉的噪音行。"""
    stripped = line.strip()
    return any(pattern in stripped for pattern in _NOISE_PATTERNS)


def parse_epoch_progress(line: str) -> tuple[int, int] | None:
    """从 ``epoch 3/120`` 这类行里解析出 (当前轮, 总轮)。

    训练脚本的逐轮回调会打印这个前缀，界面据此立刻推进进度条，
    不必等 ultralytics 写完第一轮 results.csv。
    """
    match = re.match(r"\s*epoch\s+(\d+)\s*/\s*(\d+)", line)
    if not match:
        return None
    current, total = int(match.group(1)), int(match.group(2))
    if total <= 0 or current <= 0:
        return None
    return current, total


PROGRESS_PREFIX = "[progress]"

_PROGRESS_KV = re.compile(r"(\w+)=([^\s]+)")


def parse_progress_line(line: str) -> dict | None:
    """解析训练脚本的批次进度行。

    格式：``[progress] epoch=3/120 iter=72/160 rate=1.94 elapsed=312 remain=12240``
    """
    # 不能要求标记在行首：ultralytics 的 tqdm 进度条用 \r 原地刷新、不换行，
    # 我们的进度行会粘在进度条尾部（实测 `...35.1s[progress] epoch=1/1 ...`）。
    index = line.find(PROGRESS_PREFIX)
    if index < 0:
        return None
    line = line[index:]
    info: dict = {}
    for key, value in _PROGRESS_KV.findall(line):
        if "/" in value:
            left, _, right = value.partition("/")
            try:
                info[f"{key}_current"] = int(float(left))
                info[f"{key}_total"] = int(float(right))
            except ValueError:
                continue
        else:
            try:
                info[key] = float(value)
            except ValueError:
                info[key] = value
    return info or None


def looks_like_progress_bar(line: str) -> bool:
    """是否是 tqdm 进度条的碎片（含 ``━`` 之类方块字符）。"""
    if "━" in line or "╸" in line or "╺" in line:
        return True
    stripped = line.strip()
    return stripped.endswith("s") and "it/s" in stripped


def clean_log_text(text: str) -> str:
    """把整段训练日志变成人可读的多行文本。

    训练日志里混着 tqdm 的原地刷新片段（\r 分隔，且常常没有 `━` 字符，
    只剩 `1/20 2.87G ... 43%` 这种碎片）。直接塞进文本框会被折叠成一坨。
    这里改为**白名单**：只保留明确有信息量的行，其余一律丢弃。
    完整原始日志仍在 outputs/runs/<实验>/train.log 里，不会丢数据。
    """
    import re

    keep_prefix = (
        "epoch ", "[", "Traceback", "RuntimeError", "ValueError", "File \"",
        "项目", "启动训练", "参数", "数据集配置", "类别数", "结构", "注意力模块数量",
        "训练日志", "设备", "优化器", "按当前数据集", "Ultralytics ",
        "optimizer:", "AMP:", "Image sizes", "Starting training", "Using ",
        "Freezing", "Transferred", "Results saved", "best.pt", "last.pt",
    )

    def informative(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith(keep_prefix):
            return True
        # 扫描/标签异常这类提示保留，逐张图的重复信息丢弃
        if ("Scanning" in stripped or "corrupt" in stripped
                or "duplicate labels" in stripped[:12]):
            return "Scanning" in stripped
        return False

    out = []
    for raw in re.split(r"[\r\n]", strip_ansi(text)):
        line = raw.rstrip()
        if informative(line):
            out.append(line)
    return "\n".join(out)
