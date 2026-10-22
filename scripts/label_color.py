# -*- coding: utf-8 -*-
"""全量毛色标注。

对每个检测框执行「分割 → 取色 → 分档」，输出逐目标的毛色标签与特征值。
标签附带置信度，低置信度样本可被下游过滤，不要当成硬标签使用。

用法::

    python scripts/label_color.py
    python scripts/label_color.py --limit 200      # 先小批量试跑
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lps import config  # noqa: E402
from lps.data import color_labeler as cl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="毛色自动标注")
    p.add_argument("--limit", type=int, default=0, help="每个划分最多处理多少张图")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else (config.OUTPUT_DIR / "attributes" / "color_labels.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[list] = []
    dist: Counter = Counter()
    conf_by_label: dict[str, list[float]] = {}
    total = 0
    skipped_img = 0

    for split in ("train", "valid", "test"):
        img_dir = config.DATA_ROOT / split / "images"
        lab_dir = config.DATA_ROOT / split / "labels"
        if not lab_dir.exists():
            continue
        labels = sorted(lab_dir.glob("*.txt"))
        if args.limit:
            labels = labels[: args.limit]

        for i, lab in enumerate(labels):
            match = list(img_dir.glob(f"{lab.stem}.*"))
            if not match:
                skipped_img += 1
                continue
            image = cl.imread_unicode(match[0])
            if image is None:
                skipped_img += 1
                continue

            boxes = []
            for line in lab.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                boxes.append((int(float(parts[0])), tuple(float(v) for v in parts[1:])))

            for cls_id, box in boxes:
                f = cl.analyze_box(image, box)
                total += 1
                dist[f.label] += 1
                conf_by_label.setdefault(f.label, []).append(f.confidence)
                rows_out.append([
                    split, lab.stem, cls_id,
                    f"{box[0]:.6f}", f"{box[1]:.6f}", f"{box[2]:.6f}", f"{box[3]:.6f}",
                    f.label, cl.COLOR_ZH.get(f.label, f.label),
                    f"{f.confidence:.3f}", f"{f.l_med:.2f}", f"{f.a_med:.2f}",
                    f"{f.b_med:.2f}", f"{f.chroma:.2f}", f"{f.bimodal:.3f}",
                    f"{f.fg_ratio:.3f}",
                ])
            if (i + 1) % 200 == 0:
                print(f"  {split}: {i + 1}/{len(labels)} 张，累计 {total} 个目标")

    header = ["split", "image", "cls_id", "x_center", "y_center", "width", "height",
              "label", "label_zh", "confidence", "l_med", "a_med", "b_med",
              "chroma", "bimodal", "fg_ratio"]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)

    print(f"\n共标注 {total} 个目标，跳过 {skipped_img} 张图")
    print("毛色分布：")
    for label, n in dist.most_common():
        confs = conf_by_label[label]
        mean_conf = sum(confs) / len(confs)
        print(f"  {cl.COLOR_ZH.get(label, label):4s} {label:6s} {n:5d} ({n / total * 100:5.1f}%)  "
              f"平均置信度 {mean_conf:.2f}")

    low = sum(1 for r in rows_out if float(r[9]) < 0.45)
    print(f"低置信度（<0.45）{low} 个，占 {low / total * 100:.1f}%，建议下游过滤")
    print(f"结果: {out_path}")


if __name__ == "__main__":
    main()
