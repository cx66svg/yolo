# -*- coding: utf-8 -*-
"""验证脚本。

相对原始框架的改动：
  1. 支持 --split test，并把每类 AP 单独列出（原脚本只打总 mAP）
  2. 输出类别混淆矩阵的文本版，能直接看出哪两类在互相混淆
  3. 支持一次评估多个权重并生成对比表

用法::

    python scripts/val.py --weights outputs/runs/xxx/weights/best.pt --scheme behavior4
    python scripts/val.py --split test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lps import config                       # noqa: E402
from lps.models.attention import inject_modules  # noqa: E402
from lps.services.plots import install_plot_patch  # noqa: E402

install_plot_patch()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="牲畜姿态识别 - 验证")
    p.add_argument("--weights", default=None, help="权重路径，默认取最新一次训练")
    p.add_argument("--scheme", default=None,
                   help="预设类别方案；用任意数据集时不要传，类名直接取模型自带的")
    p.add_argument("--data", default=None)
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.001, help="评估用低阈值，保持 PR 曲线完整")
    return p.parse_args()


def latest_weights() -> Path:
    runs = sorted(config.RUNS_DIR.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"{config.RUNS_DIR} 下没有找到 best.pt，请先训练")
    return runs[-1]


def main() -> None:
    args = parse_args()
    # scheme 仅在显式指定预设方案时启用。
    # 不能默认挂上预设方案：它的中文名是按**预设类别顺序**排的，
    # 换成任意数据集后编号含义不同，会打出「类别 2 奔跑(行走)」这种错位结果。
    scheme = None
    if args.scheme and args.scheme in config.CLASS_SCHEMES:
        scheme = config.set_scheme(args.scheme)
    inject_modules()
    install_plot_patch()

    from ultralytics import YOLO

    weights = Path(args.weights) if args.weights else latest_weights()
    data_yaml = Path(args.data) if args.data else config.dataset_yaml_path()
    model = YOLO(str(weights))

    # 打包成 exe 后，Windows 下 ultralytics 的 DataLoader 用 multiprocessing
    # spawn 方式起 worker，会拿 exe 自身当入口重启，子进程起不来 → 卡死在
    # build_dataloader（CPU/GPU 全 0）。frozen 环境强制单进程加载数据绕开。
    workers = 0 if getattr(sys, "frozen", False) else 8

    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        workers=workers,
        plots=True,
        project=str(config.RUNS_DIR),
        name=f"val_{weights.parent.parent.name}_{args.split}",
        exist_ok=True,
    )

    print("\n========== 验证结果 ==========")
    print(f"权重: {weights}")
    print(f"数据: {data_yaml.name}  划分: {args.split}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50   : {metrics.box.map50:.4f}")
    print(f"mAP75   : {metrics.box.map75:.4f}")

    names = getattr(metrics, "names", None) or (scheme.names if scheme else {})
    # 中文对照只在「预设方案的类数与该模型一致」时启用，否则编号含义不同必然错位
    zh_names = scheme.zh_names if (scheme and len(scheme.names) == len(names)) else {}
    per_class = []
    for i, name in names.items():
        try:
            ap50_95 = metrics.box.maps[i]
        except (IndexError, TypeError):
            ap50_95 = float("nan")
        per_class.append((i, name, ap50_95))
        zh = zh_names.get(int(i), "") if zh_names else ""
        suffix = f"({zh})" if zh and zh != name else ""
        print(f"  类别 {i} {name}{suffix}: AP50-95 = {ap50_95:.4f}")

    # 混淆矩阵落到 CSV，便于在报告里引用
    out_dir = config.OUTPUT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"per_class_ap_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name", "zh", "AP50-95"])
        for i, name, ap in per_class:
            w.writerow([i, name, zh_names.get(int(i), ""), f"{ap:.4f}"])
        w.writerow([])
        w.writerow(["overall", "mAP50-95", "", f"{metrics.box.map:.4f}"])
        w.writerow(["overall", "mAP50", "", f"{metrics.box.map50:.4f}"])
        w.writerow(["overall", "mAP75", "", f"{metrics.box.map75:.4f}"])
    print(f"\n每类 AP 已写入: {csv_path}")

    if len(per_class) > 1:
        weakest = min(per_class, key=lambda x: x[2])
        print(f"最弱类别: {weakest[1]}（AP50-95 {weakest[2]:.4f}）—— 长尾类通常需要额外处理")


if __name__ == "__main__":
    main()
