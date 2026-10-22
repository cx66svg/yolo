"""毛色属性自动标注原型（验证用）。

目的：验证能否从现有检测框直接推导毛色，从而免去人工标注。
做法：
  1. 取框并向外扩 8%，保证包含完整个体
  2. 在 HSV 空间剔除草地绿、天空高亮等背景像素
  3. 对剩余像素用 K-means 取主色，按亮度/饱和度/色相判定白/黑/棕/花
输出一张对照图，人工目检判定是否可用。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(r"C:\Users\Administrator\Desktop\山羊检测\sheep_behavior_yolov8(1)\sheep_behavior_yolov8\data")
OUT = Path(__file__).resolve().parent.parent / "outputs" / "color_probe"
CLASS_EN = {0: "FEED", 1: "REST", 2: "RUN", 3: "WALK"}


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread 在 Windows 下不支持中文路径，用 imdecode 绕开。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 92) -> bool:
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image, [cv2.IMWRITE_JPEG_QUALITY, quality]
                           if ext.lower() in (".jpg", ".jpeg") else [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def mask_background(hsv: np.ndarray) -> np.ndarray:
    """返回前景掩码：去掉草地绿与极暗阴影。"""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (h > 30) & (h < 90) & (s > 45) & (v > 40)
    very_dark = v < 25
    return ~(grass | very_dark)


def dominant_color(bgr: np.ndarray) -> tuple[str, np.ndarray]:
    """返回 (毛色标签, 代表色 BGR)。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    fg = mask_background(hsv)
    if fg.sum() < 50:
        fg = np.ones(hsv.shape[:2], bool)

    pixels = bgr[fg].reshape(-1, 3).astype(np.float32)
    if len(pixels) > 4000:
        idx = np.random.default_rng(0).choice(len(pixels), 4000, replace=False)
        pixels = pixels[idx]

    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    main = centers[order[0]].astype(np.uint8)

    # 逐簇判定，再看主色与次色构成
    def classify(c: np.ndarray) -> str:
        hh, ss, vv = cv2.cvtColor(c.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        if vv < 70:
            return "BLACK"
        if vv > 135 and ss < 55:
            return "WHITE"
        if 8 <= hh <= 30 and ss > 50:
            return "BROWN"
        return "WHITE" if vv > 110 else "黑"

    tags = [classify(centers[i]) for i in order]
    ratio = counts[order] / counts.sum()

    if ratio[0] < 0.55 and len(set(tags[:2])) > 1 and ratio[1] > 0.2:
        return "MIXED", main
    return tags[0], main


def main(n_per_class: int = 6) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(7)

    label_files = sorted((SRC / "train" / "labels").glob("*.txt"))
    by_class: dict[int, list[Path]] = {}
    for lf in label_files:
        for line in lf.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                by_class.setdefault(int(float(p[0])), []).append(lf)
                break

    tiles = []
    for cls, files in sorted(by_class.items()):
        for lf in random.sample(files, min(n_per_class, len(files))):
            img_path = SRC / "train" / "images" / f"{lf.stem}.jpg"
            if not img_path.exists():
                img_path = next((SRC / "train" / "images").glob(f"{lf.stem}.*"), None)
            if img_path is None:
                continue
            img = imread_unicode(img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            first = lf.read_text(encoding="utf-8").splitlines()[0].split()
            _, cx, cy, bw, bh = (float(v) for v in first)
            x1 = int((cx - bw / 2) * w_img); x2 = int((cx + bw / 2) * w_img)
            y1 = int((cy - bh / 2) * h_img); y2 = int((cy + bh / 2) * h_img)
            mx, my = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
            x1, y1 = max(0, x1 - mx), max(0, y1 - my)
            x2, y2 = min(w_img, x2 + mx), min(h_img, y2 + my)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            tag, rep = dominant_color(crop)
            tile = cv2.resize(crop, (160, 160))
            cv2.rectangle(tile, (0, 0), (159, 159), tuple(int(v) for v in rep), 3)
            cv2.putText(tile, f"{CLASS_EN.get(cls,'?')} {tag}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(tile, f"{CLASS_EN.get(cls,'?')} {tag}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            tiles.append(tile)

    if not tiles:
        print("没有取到样本")
        return

    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * 160, cols * 160, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 160:(r + 1) * 160, c * 160:(c + 1) * 160] = t

    dst = OUT / "color_probe.jpg"
    cv2.imwrite(str(dst), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"样本 {len(tiles)} 个，对照图: {dst}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
