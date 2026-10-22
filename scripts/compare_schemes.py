# -*- coding: utf-8 -*-
"""跨类别方案的公平对比。

为什么不能直接比总 mAP：
    4 类方案的 mAP 是在 4 个类别上平均，8 类方案是在 8 个类别上平均，
    类别定义不同，数字不可直接比较。

这里改成「行为层面」的比较：
    1. 把 8 类模型输出的 (采食正面/采食侧面) 合并回「采食」，
       其余行为同理，映射规则见 lps.data.attributes 中的 BEHAVIOR_OF
    2. 对所有模型统一按 IoU≥0.5 与原始 4 类标注做贪心匹配
    3. 统计每个行为的精确率、召回率、F1

这样两个模型面对同一批真实标注、同一个判定标准，结果可比。

用法::

    python scripts/compare_schemes.py --a outputs/runs/cmp_behavior4/weights/best.pt \
                                      --b outputs/runs/cmp_pose8/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lps import config  # noqa: E402

BEHAVIORS = {0: "采食", 1: "休息", 2: "奔跑", 3: "行走"}

# 行为名 -> 原始标注里的编号。
# 必须按**类名**映射，不能按模型输出的编号直接当行为编号：
# behavior4 方案派生时重排过（2=walking、3=running），而原始标注是
# 2=奔跑、3=行走，直接拿编号比较会把奔跑与行走对调，两项一起失真。
NAME_TO_BEHAVIOR = {"feeding": 0, "resting": 1, "running": 2, "walking": 3,
                    "eat": 0, "rest": 1, "run": 2, "walk": 3}


def behavior_of(cls_id: int, names: dict) -> int:
    """把模型输出的类别编号换算成原始标注体系下的行为编号。

    pose_view8 的类名形如 feeding_front / walking_side，去掉下划线后缀
    即得行为名，所以同一套逻辑两种方案都适用。
    """
    raw = str(names.get(cls_id, cls_id)).strip().lower()
    base = raw.split("_")[0]
    return NAME_TO_BEHAVIOR.get(base, cls_id)


def xywh_to_xyxy(x, y, w, h, W, H):
    return (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_ground_truth(split: str) -> dict[str, list[tuple]]:
    """读取原始 4 类标注，返回 {图像名: [(行为编号, xyxy 归一化)]}。"""
    import cv2
    import numpy as np

    img_dir = config.DATA_ROOT / split / "images"
    lab_dir = config.DATA_ROOT / split / "labels"
    truth: dict[str, list[tuple]] = {}
    for lab in sorted(lab_dir.glob("*.txt")):
        img_path = img_dir / f"{lab.stem}.jpg"
        if not img_path.exists():
            matches = list(img_dir.glob(f"{lab.stem}.*"))
            if not matches:
                continue
            img_path = matches[0]
        buf = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if img is None:
            continue
        H, W = img.shape[:2]
        boxes = []
        for line in lab.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
            boxes.append((cid, xywh_to_xyxy(x, y, w, h, W, H)))
        truth[img_path.name] = boxes
    return truth


def evaluate(weights: Path, truth: dict, split: str, conf: float = 0.25) -> dict:
    from ultralytics import YOLO

    from lps.models.attention import inject_modules

    inject_modules()
    model = YOLO(str(weights))
    names_map = dict(model.names)

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    img_dir = config.DATA_ROOT / split / "images"
    names = list(truth)
    for i in range(0, len(names), 16):
        batch = [str(img_dir / n) for n in names[i:i + 16]]
        results = model.predict(batch, conf=conf, iou=0.7, verbose=False, device="0")
        for name, r in zip(names[i:i + 16], results):
            gts = truth[name]
            preds = []
            for box in r.boxes:
                cid = behavior_of(int(box.cls.item()), names_map)
                preds.append((cid, tuple(box.xyxy[0].tolist()), float(box.conf.item())))
            preds.sort(key=lambda p: -p[2])

            used = set()
            for cid, pbox, _ in preds:
                best_i, best_v = -1, 0.0
                for gi, (gcid, gbox) in enumerate(gts):
                    if gi in used:
                        continue
                    v = iou(pbox, gbox)
                    if v > best_v:
                        best_i, best_v = gi, v
                if best_v >= 0.5 and best_i >= 0:
                    used.add(best_i)
                    gcid = gts[best_i][0]
                    if gcid == cid:
                        stats[cid]["tp"] += 1
                    else:
                        stats[cid]["fp"] += 1
                        stats[gcid]["fn"] += 1
                else:
                    stats[cid]["fp"] += 1
            for gi, (gcid, _) in enumerate(gts):
                if gi not in used:
                    stats[gcid]["fn"] += 1

    out = {}
    for cid in BEHAVIORS:
        s = stats[cid]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[cid] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="跨方案行为级对比")
    ap.add_argument("--a", required=True, help="方案 A 权重")
    ap.add_argument("--b", required=True, help="方案 B 权重")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    print(f"读取标注（{args.split}）…")
    truth = load_ground_truth(args.split)
    print(f"  {len(truth)} 张图")

    print(f"评估 {args.label_a} …")
    res_a = evaluate(Path(args.a), truth, args.split, args.conf)
    print(f"评估 {args.label_b} …")
    res_b = evaluate(Path(args.b), truth, args.split, args.conf)

    header = f"{'行为':6s}{'方案':10s}{'精确率':>9s}{'召回率':>9s}{'F1':>9s}"
    print("\n" + header)
    print("-" * len(header))
    for cid, zh in BEHAVIORS.items():
        for label, res in ((args.label_a, res_a), (args.label_b, res_b)):
            m = res[cid]
            print(f"{zh:6s}{label:10s}{m['precision']:9.3f}{m['recall']:9.3f}{m['f1']:9.3f}")
        print()

    def macro(res):
        return sum(res[c]["f1"] for c in BEHAVIORS) / len(BEHAVIORS)

    fa, fb = macro(res_a), macro(res_b)
    print(f"宏平均 F1：{args.label_a} = {fa:.4f}   {args.label_b} = {fb:.4f}")
    winner = args.label_a if fa > fb else args.label_b
    print(f"行为层面表现更好：{winner}（差 {abs(fa - fb):.4f}）")

    out = config.OUTPUT_DIR / "reports" / "scheme_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["行为", "方案", "精确率", "召回率", "F1", "TP", "FP", "FN"])
        for cid, zh in BEHAVIORS.items():
            for label, res in ((args.label_a, res_a), (args.label_b, res_b)):
                m = res[cid]
                w.writerow([zh, label, f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                            f"{m['f1']:.4f}", m["tp"], m["fp"], m["fn"]])
    print(f"\n对比表已写入 {out}")


if __name__ == "__main__":
    main()
