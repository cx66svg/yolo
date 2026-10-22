# -*- coding: utf-8 -*-
"""毛色标注原型（第二版）：分割后取色，输出对照图供人工核验。

每个小格显示的是**分割后的结果**（背景被涂成中灰），
可以直接看出 GrabCut 有没有把背景误当成羊。
左上角文字为：判定结果 + 置信度 + 明度L + 黄度b + 色度c + 明度std。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lps import config  # noqa: E402
from lps.data import color_labeler as cl  # noqa: E402

BG_GREY = (90, 90, 90)


def tile(image: np.ndarray, box, idx: int, size: int = 168) -> tuple[np.ndarray, str]:
    crop, _ = cl.crop_box(image, box)
    if crop.size == 0:
        return np.full((size, size, 3), BG_GREY, np.uint8), "empty"

    fg = cl.segment(crop)
    feats = cl.classify(cl.extract_features(crop, fg))

    shown = crop.copy()
    shown[~fg] = BG_GREY
    shown = cv2.resize(shown, (size, size), interpolation=cv2.INTER_AREA)

    border = tuple(int(cl.COLOR_HEX[feats.label].lstrip("#")[i:i + 2], 16) for i in (4, 2, 0))
    cv2.rectangle(shown, (0, 0), (size - 1, size - 1), border, 4)

    text = f"{feats.label} {feats.confidence:.2f}"
    detail = f"L{feats.l_med:.0f} b{feats.b_med:+.0f} c{feats.chroma:.0f} bi{feats.bimodal:.2f}"
    cv2.putText(shown, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3)
    cv2.putText(shown, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.putText(shown, detail, (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
    cv2.putText(shown, detail, (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return shown, feats.label


def main(n: int = 32) -> None:
    out_dir = config.OUTPUT_DIR / "color_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(11)

    img_dir = config.DATA_ROOT / "train" / "images"
    lab_dir = config.DATA_ROOT / "train" / "labels"
    labels = sorted(lab_dir.glob("*.txt"))
    random.shuffle(labels)

    tiles, stats = [], {}
    for lab in labels:
        if len(tiles) >= n:
            break
        img_path = img_dir / f"{lab.stem}.jpg"
        if not img_path.exists():
            match = list(img_dir.glob(f"{lab.stem}.*"))
            if not match:
                continue
            img_path = match[0]
        image = cl.imread_unicode(img_path)
        if image is None:
            continue
        rows = [l.split() for l in lab.read_text(encoding="utf-8").splitlines() if len(l.split()) == 5]
        if not rows:
            continue
        # 取该图最大的框，避免小目标分割不稳定
        best = max(rows, key=lambda r: float(r[3]) * float(r[4]))
        box = tuple(float(v) for v in best[1:])
        t, label = tile(image, box, len(tiles))
        tiles.append(t)
        stats[label] = stats.get(label, 0) + 1

    cols = 8
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows_n * 168, cols * 168, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 168:(r + 1) * 168, c * 168:(c + 1) * 168] = t

    dst = out_dir / "color_probe_v2.jpg"
    cl.imwrite_unicode(dst, sheet)
    print(f"样本 {len(tiles)} 个，分布 {stats}")
    print(f"对照图: {dst}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
