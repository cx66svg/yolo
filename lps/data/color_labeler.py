"""毛色属性自动标注。

前一次尝试（直接在裁剪图上按 HSV 阈值判定）准确率只有三成多，原因是
背景的干土、枯草、阴影与羊毛同色系，简单阈值无法区分前景与背景。

本模块改成三步走：
    1. 分割：以检测框为初始矩形跑 GrabCut，把羊从背景里抠出来
    2. 特征：只在分割得到的前景像素上统计 Lab 空间分布
    3. 判定：按明度 L、色度 chroma、黄度 b* 与明度离散度分档

判定规则集中在这里，便于拿真实样本反复校准；不要散落到调用方。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 毛色类别（与 class_scheme 中的属性取值保持一致）
WHITE = "white"
CREAM = "cream"
TAN = "tan"
BROWN = "brown"
GREY = "grey"
DARK = "dark"
MIXED = "mixed"

COLOR_ZH = {
    WHITE: "白",
    CREAM: "米白",
    TAN: "棕黄",
    BROWN: "褐色",
    GREY: "灰",
    DARK: "深褐",
    MIXED: "花",
}

# 色板，用于界面与对照图
COLOR_HEX = {
    WHITE: "#F2F2F2",
    CREAM: "#E8DCC8",
    TAN: "#C89B6A",
    BROWN: "#8B5E3C",
    GREY: "#A8A8A0",
    DARK: "#4A3B32",
    MIXED: "#B08968",
}


@dataclass
class ColorFeatures:
    l_med: float          # Lab 明度中位数（0-100）
    a_med: float          # 红绿轴
    b_med: float          # 黄蓝轴
    chroma: float         # sqrt(a^2 + b^2)
    l_std: float          # 明度标准差
    fg_ratio: float       # 前景像素占裁剪框比例
    label: str = ""
    confidence: float = 0.0
    bimodal: float = 0.0  # 双峰程度，见 bimodal_ratio

    def as_dict(self) -> dict:
        return {
            "l_med": round(self.l_med, 2), "a_med": round(self.a_med, 2),
            "b_med": round(self.b_med, 2), "chroma": round(self.chroma, 2),
            "l_std": round(self.l_std, 2), "fg_ratio": round(self.fg_ratio, 3),
            "bimodal": round(self.bimodal, 3),
            "label": self.label, "confidence": round(self.confidence, 3),
        }


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """cv2.imread 不支持中文路径。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 92) -> bool:
    ext = path.suffix or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image, params)
    if ok:
        buf.tofile(str(path))
    return ok


def crop_box(image: np.ndarray, box: tuple[float, float, float, float],
             margin: float = 0.06) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """按归一化中心框裁剪，返回裁剪图与在图中的像素位置。"""
    h, w = image.shape[:2]
    cx, cy, bw, bh = box
    x1 = int(max(0, (cx - bw / 2) * w))
    x2 = int(min(w, (cx + bw / 2) * w))
    y1 = int(max(0, (cy - bh / 2) * h))
    y2 = int(min(h, (cy + bh / 2) * h))
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def segment(crop: np.ndarray, inset: float = 0.10) -> np.ndarray:
    """用 GrabCut 从裁剪图里抠出目标，返回布尔前景掩码。"""
    h, w = crop.shape[:2]
    if h < 12 or w < 12:
        return np.ones((h, w), bool)

    mask = np.zeros((h, w), np.uint8)
    dx, dy = int(w * inset), int(h * inset)
    rect = (dx, dy, max(1, w - 2 * dx), max(1, h - 2 * dy))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    except cv2.error:
        return np.ones((h, w), bool)

    # GrabCut 偶尔会退化成全背景，此时回退到矩形内区域
    if fg.mean() < 0.08:
        fg = np.zeros((h, w), bool)
        fg[dy:h - dy, dx:w - dx] = True
    return fg


def drop_grass_pixels(crop: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """从前景里剔除草叶像素。

    GrabCut 有时会把目标脚下或身后的草一起判成前景，草是黄绿色，
    会同时拉低 L、抬高 b*，把羊判成"褐色/花色"。这里按色相与饱和度
    排除草绿（白羊、棕羊的色相都在橙褐段，不会被误伤）。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (h > 30) & (h < 95) & (s > 70) & (v > 30)
    cleaned = fg & ~grass
    # 剔得太狠说明前景本来就不多，退回原掩码避免统计样本过少
    if cleaned.sum() < fg.sum() * 0.35:
        return fg
    return cleaned


def bimodal_ratio(L: np.ndarray, mask: np.ndarray) -> float:
    """明度的双峰程度：暗部与亮部像素占比较小者。

    纯色羊受光照影响也会有大标准差，但只要不是真的黑白相间，
    就不会同时出现大量"暗"像素和大量"亮"像素。用两个区间的最小占比
    作为判据，比直接看标准差稳得多。
    """
    values = L[mask]
    if values.size == 0:
        return 0.0
    dark = float((values < 40).mean())
    bright = float((values > 68).mean())
    return min(dark, bright)


def extract_features(crop: np.ndarray, fg: np.ndarray) -> ColorFeatures:
    """在前景像素上统计 Lab 分布。"""
    fg = drop_grass_pixels(crop, fg)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV 的 Lab：L 0-255, a/b 0-255，转换到常规量纲便于设阈值
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    if fg.sum() < 30:
        fg = np.ones(crop.shape[:2], bool)

    l_med = float(np.median(L[fg]))
    a_med = float(np.median(a[fg]))
    b_med = float(np.median(b[fg]))
    chroma = float(np.median(np.sqrt(a[fg] ** 2 + b[fg] ** 2)))
    l_std = float(np.std(L[fg]))
    return ColorFeatures(
        l_med=l_med, a_med=a_med, b_med=b_med, chroma=chroma,
        l_std=l_std, fg_ratio=float(fg.mean()),
        bimodal=bimodal_ratio(L, fg),
    )


def classify(f: ColorFeatures) -> ColorFeatures:
    """按 Lab 统计量分档。阈值由真实样本校准得到，改动需重新验证。

    判定顺序：
      1. 真双峰 → 花色
      2. 明度高档里，色度极高（c≥26）的其实是橙棕色羊在强光下被拉亮，
         必须先截走，否则会被判成米白（实测这类错误占误判的三分之一）
      3. 明度中档里，色度极低（c<8）的是灰羊，不能因为 b* 不够黄就归到米白
    """
    L, b, c = f.l_med, f.b_med, f.chroma

    if f.bimodal >= 0.22 and f.l_std >= 14:
        f.label, f.confidence = MIXED, min(1.0, f.bimodal / 0.35)
    elif L >= 70:
        f.label = TAN if c >= 26 else (WHITE if c < 13 else CREAM)
        f.confidence = 0.82
    elif L >= 62:
        f.label = TAN if c >= 20 else CREAM
        f.confidence = 0.75
    elif L >= 49:
        if c < 8:
            f.label = GREY
        elif b >= 12:
            f.label = TAN
        else:
            f.label = CREAM
        f.confidence = 0.74
    elif L >= 37:
        f.label = BROWN if b >= 8 else GREY
        f.confidence = 0.72
    elif L >= 27:
        f.label = GREY if c < 9 else DARK
        f.confidence = 0.60
    else:
        f.label, f.confidence = DARK, 0.78

    # 前景占比过低说明分割没抠准（目标太小或与背景粘连），降低可信度
    if f.fg_ratio < 0.15:
        f.confidence *= 0.5
    return f


def analyze_box(image: np.ndarray, box: tuple[float, float, float, float]) -> ColorFeatures:
    """对单个目标给出毛色判定。"""
    crop, _ = crop_box(image, box)
    if crop.size == 0:
        return ColorFeatures(0, 0, 0, 0, 0, 0, WHITE, 0.0)
    fg = segment(crop)
    return classify(extract_features(crop, fg))
