# -*- coding: utf-8 -*-
"""推理脚本。

支持单张图片、图片目录、视频文件与摄像头。结果保存为标注图、
检测明细 CSV 与汇总 JSON，供界面展示与报告导出复用。

用法::

    python scripts/predict.py --source img.jpg
    python scripts/predict.py --source video.mp4 --conf 0.3
    python scripts/predict.py --source 0 --max-frames 300
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lps import config                       # noqa: E402
from lps.core import Detector                # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".flv"}


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread 不支持中文路径，走 imdecode。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))
    return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="牲畜姿态识别 - 推理")
    p.add_argument("--weights", default=None, help="权重，默认取最新 best.pt")
    p.add_argument("--source", required=True, help="图片 / 目录 / 视频 / 摄像头编号")
    p.add_argument("--scheme", default=None,
                   help="预设类别方案；用任意数据集时不要传，类名取模型自带的")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="run")
    p.add_argument("--max-frames", type=int, default=0, help="视频/摄像头最大帧数，0 表示不限制")
    return p.parse_args()


def latest_weights() -> Path:
    runs = sorted(config.RUNS_DIR.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError("没有找到已训练的 best.pt，请先训练")
    return runs[-1]


def main() -> None:
    args = parse_args()
    # 任意数据集场景下没有预设方案，Detector 会退回用模型自带类名
    scheme = None
    if args.scheme and args.scheme in config.CLASS_SCHEMES:
        scheme = config.set_scheme(args.scheme)

    weights = Path(args.weights) if args.weights else latest_weights()
    out_dir = config.OUTPUT_DIR / "predict" / args.name
    img_dir = out_dir / "annotated"
    img_dir.mkdir(parents=True, exist_ok=True)
    # 清空旧标注图，避免本次 0 结果时界面还显示上次的图
    for old in img_dir.iterdir():
        if old.is_file():
            old.unlink()
    for old in (out_dir / "summary.json", out_dir / "detections.csv"):
        if old.exists():
            old.unlink()

    detector = Detector(weights, scheme=scheme, device=args.device,
                        conf=args.conf, iou=args.iou, imgsz=args.imgsz)

    rows: list[list] = []
    frames_summary: list[dict] = []
    source = args.source

    if source.isdigit():                      # 摄像头
        cap = cv2.VideoCapture(int(source))
        index = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            result, annotated = detector.predict(frame)
            dst = img_dir / f"frame_{index:05d}.jpg"
            imwrite_unicode(dst, annotated)
            frames_summary.append(_frame_record(index, result))
            rows.extend(_rows_for(index, result))
            index += 1
            if args.max_frames and index >= args.max_frames:
                break
        cap.release()
        print(f"摄像头采集 {index} 帧")

    else:
        path = Path(source)
        if path.is_dir():
            # YOLO 数据集常见结构：test/images/；也支持图片直接放在目录里
            image_root = path / "images" if (path / "images").is_dir() else path
            images = sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_EXT)
            for i, img in enumerate(images):
                frame = imread_unicode(img)
                if frame is None:
                    continue
                result, annotated = detector.predict(frame)
                imwrite_unicode(img_dir / f"{img.stem}.jpg", annotated)
                frames_summary.append(_frame_record(i, result, img.name))
                rows.extend(_rows_for(img.name, result))
            print(f"目录共 {len(images)} 张，已处理 {len(frames_summary)} 张")
            if not images:
                print(f"警告：{image_root} 中没有支持的图片格式 {IMAGE_EXT}")

        elif path.suffix.lower() in VIDEO_EXT:
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            index = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                result, annotated = detector.predict(frame)
                writer.write(annotated)
                frames_summary.append(_frame_record(index, result))
                rows.extend(_rows_for(index, result))
                index += 1
                if args.max_frames and index >= args.max_frames:
                    break
            writer.release()
            cap.release()
            print(f"视频处理 {index} 帧，结果：{out_dir / 'annotated.mp4'}")

        elif path.suffix.lower() in IMAGE_EXT:
            frame = imread_unicode(path)
            if frame is None:
                raise SystemExit(f"无法读取图片：{path}")
            result, annotated = detector.predict(frame)
            imwrite_unicode(img_dir / f"{path.stem}.jpg", annotated)
            frames_summary.append(_frame_record(0, result, path.name))
            rows.extend(_rows_for(path.name, result))
            print(f"检测到 {len(result.detections)} 个目标")
        else:
            raise SystemExit(f"不支持的输入类型：{path.suffix}")

    # 汇总
    counts: dict[str, int] = {}
    for rec in frames_summary:
        for k, v in rec.get("counts", {}).items():
            counts[k] = counts.get(k, 0) + v
    summary = {
        "weights": str(weights),
        "source": str(source),
        "frames": len(frames_summary),
        "total_objects": sum(counts.values()),
        "class_counts": counts,
        "per_frame": frames_summary[:2000],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["frame", "class", "zh", "conf", "x1", "y1", "x2", "y2",
                    "zone", "scale", "occlusion", "crowd"])
        w.writerows(rows)

    print("\n========== 检测汇总 ==========")
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {n}")
    print(f"  合计 {summary['total_objects']} 个目标，{summary['frames']} 帧")
    print(f"结果目录: {out_dir}")


def _frame_record(index: int, result, name: str = "") -> dict:
    return {
        "index": index,
        "image": name,
        "counts": result.counts(),
        "zone": result.attribute_summary("zone"),
        "scale": result.attribute_summary("scale"),
        "occlusion": result.attribute_summary("occlusion"),
        "crowd": result.attribute_summary("crowd"),
    }


def _rows_for(frame_key, result) -> list[list]:
    out = []
    for d in result.detections:
        out.append([frame_key, d.name, d.zh, f"{d.conf:.4f}",
                    f"{d.box[0]:.1f}", f"{d.box[1]:.1f}",
                    f"{d.box[2]:.1f}", f"{d.box[3]:.1f}",
                    d.zone, d.scale, d.occlusion, d.crowd])
    return out


if __name__ == "__main__":
    main()
