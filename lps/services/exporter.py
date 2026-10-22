"""结果导出。

把训练指标、逐类精度、检测明细、目标属性统计整理成可直接放进报告的文件：
    - Excel 多工作表（需要 openpyxl，缺失时降级为 CSV）
    - Markdown 摘要（便于粘进 Word，再手动排版）
    - 图表 PNG（复用 services.plots 的样式）

设计原则：所有导出都从磁盘上已有的产物读，不重新推理，
因此导出动作本身很轻，不会因为模型在训练而失败。
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .. import config


@dataclass
class ExportResult:
    files: List[Path]

    def summary(self) -> str:
        if not self.files:
            return "没有可导出的内容"
        return "已导出:\n" + "\n".join(f"  {p}" for p in self.files)


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None


def collect_training(run_dir: Path) -> Optional[pd.DataFrame]:
    """读取某次训练的 results.csv。"""
    return _read_csv(Path(run_dir) / "results.csv")


def collect_per_class(split: str = "test") -> Optional[pd.DataFrame]:
    """读取逐类 AP 表（由 val.py 生成）。"""
    return _read_csv(config.OUTPUT_DIR / "reports" / f"per_class_ap_{split}.csv")


def collect_detections(name: str = "gui") -> Optional[pd.DataFrame]:
    """读取某次推理的检测明细。"""
    return _read_csv(config.OUTPUT_DIR / "predict" / name / "detections.csv")


def collect_attributes() -> Optional[pd.DataFrame]:
    """读取目标属性表（区域/尺度/遮挡/密度）。"""
    return _read_csv(config.OUTPUT_DIR / "attributes" / "object_attributes.csv")


def collect_colors() -> Optional[pd.DataFrame]:
    """读取毛色标签表。"""
    return _read_csv(config.OUTPUT_DIR / "attributes" / "color_labels.csv")


def build_summary_markdown(run_dir: Optional[Path] = None,
                           split: str = "test") -> str:
    """生成一段可直接粘进报告的 Markdown 摘要。"""
    lines = ["# 实验结果摘要", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

    per_class = collect_per_class(split)
    if per_class is not None and not per_class.empty:
        overall = per_class[per_class["class_id"] == "overall"]
        table = per_class[per_class["class_id"] != "overall"]
        lines += [f"## {split} 集总体指标", ""]
        if not overall.empty:
            for _, row in overall.iterrows():
                lines.append(f"- {row['name']}：{row['AP50-95']}")
        lines += ["", "## 逐类 AP@0.5:0.95", "",
                  "| 类别 | 中文名 | AP50-95 |", "| --- | --- | --- |"]
        for _, row in table.iterrows():
            lines.append(f"| {row['name']} | {row.get('zh', '')} | {row['AP50-95']} |")
        lines.append("")

    if run_dir:
        frame = collect_training(run_dir)
        if frame is not None and not frame.empty:
            frame.columns = [c.strip() for c in frame.columns]
            best_idx = frame["metrics/mAP50-95(B)"].astype(float).idxmax()
            best = frame.loc[best_idx]
            lines += ["## 训练概况", "",
                      f"- 总轮数：{len(frame)}",
                      f"- 最佳轮次：{int(best['epoch']) + 1}",
                      f"- 最佳 mAP@0.5：{float(best['metrics/mAP50(B)']):.4f}",
                      f"- 最佳 mAP@0.5:0.95：{float(best['metrics/mAP50-95(B)']):.4f}",
                      ""]

    det = collect_detections()
    if det is not None and not det.empty:
        lines += ["## 检测统计", "",
                  f"- 目标总数：{len(det)}",
                  f"- 涉及帧数：{det['frame'].nunique()}"]
        if "zh" in det.columns:
            counts = det["zh"].value_counts()
            lines.append("- 类别分布：" + "，".join(f"{k} {v}" for k, v in counts.items()))
        lines.append("")
    return "\n".join(lines)


def export_all(run_dir: Optional[Path] = None, split: str = "test",
               out_dir: Optional[Path] = None) -> ExportResult:
    """导出全部可用结果：Excel（或多 CSV）+ Markdown + 图表。"""
    out = Path(out_dir or config.EXPORT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    files: List[Path] = []

    sheets: Dict[str, pd.DataFrame] = {}
    training = collect_training(run_dir) if run_dir else None
    if training is not None:
        training.columns = [c.strip() for c in training.columns]
        sheets["训练曲线"] = training
    per_class = collect_per_class(split)
    if per_class is not None:
        sheets[f"{split}逐类AP"] = per_class
    detections = collect_detections()
    if detections is not None:
        sheets["检测明细"] = detections
    attrs = collect_attributes()
    if attrs is not None:
        sheets["目标属性"] = attrs.head(5000)
    colors = collect_colors()
    if colors is not None:
        sheets["毛色标签"] = colors.head(5000)

    if sheets:
        xlsx = out / f"实验结果_{stamp}.xlsx"
        try:
            with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                for name, frame in sheets.items():
                    frame.to_excel(writer, sheet_name=name[:31], index=False)
            files.append(xlsx)
        except (ImportError, ModuleNotFoundError):
            # 没装 openpyxl 时退化为逐表 CSV，不让导出功能失效
            for name, frame in sheets.items():
                csv_path = out / f"{name}_{stamp}.csv"
                frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
                files.append(csv_path)

    summary = build_summary_markdown(run_dir, split)
    md_path = out / f"结果摘要_{stamp}.md"
    md_path.write_text(summary, encoding="utf-8")
    files.append(md_path)

    if run_dir:
        try:
            from .plots import plot_training_results

            # 曲线图必须写进导出目录（out），不能写回 run_dir——
            # 用户选中的权重目录可能是他自己的数据，写进去会污染数据。
            png = plot_training_results(
                file=str(Path(run_dir) / "results.csv"), out_dir=out)
            if png and Path(png).exists():
                files.append(Path(png))
        except Exception:  # noqa: BLE001
            pass

    return ExportResult(files=files)


def export_detection_report(name: str = "gui") -> Optional[Path]:
    """把一次推理的统计结果写成 JSON + Markdown，供报告引用。"""
    src = config.OUTPUT_DIR / "predict" / name / "summary.json"
    if not src.exists():
        return None
    data = json.loads(src.read_text(encoding="utf-8"))
    out = config.EXPORT_DIR / f"检测报告_{name}_{_stamp()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# 检测报告", "",
             f"- 模型：{data.get('weights', '')}",
             f"- 输入：{data.get('source', '')}",
             f"- 帧数：{data.get('frames', 0)}",
             f"- 目标总数：{data.get('total_objects', 0)}", "",
             "## 类别分布", ""]
    counts = data.get("class_counts", {})
    total = sum(counts.values()) or 1
    for name_, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {name_}：{n}（{n / total * 100:.1f}%）")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def export_model(weights: Path, out_dir: Path, project=None,
                 include_dataset: bool = False) -> ExportResult:
    """一键导出**可脱离图形界面**的模型交付包。

    不只是给一个 .pt：连同训练/验证/推理脚本、源码包、依赖清单与数据配置
    一起打包。对方机器只要装了 Python 和依赖，不装本软件也能用命令行
    重新训练、验证与推理。

    目录结构::

        <项目名>_package/
        ├─ best.pt                权重
        ├─ model_card.json        类别表、类别数、来源实验、指标、用法
        ├─ data.yaml              本项目的数据集配置（绝对路径）
        ├─ requirements.txt       依赖清单
        ├─ 使用说明.md            命令与步骤
        ├─ train.py / val.py / predict.py / check_dataset.py
        ├─ lps/                   源码包（含 CBAM 自定义模块）
        ├─ scripts/               任务脚本
        └─ charts/                训练曲线与混淆矩阵
    """
    import shutil

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"权重文件不存在：{weights}")

    stem = project.name if project else weights.parent.parent.name
    stem = "".join(ch for ch in stem if ch.isalnum() or ch in "-_") or "model"
    pkg = Path(out_dir) / f"{stem}_package"
    pkg.mkdir(parents=True, exist_ok=True)

    src_root = config.PROJECT_ROOT
    files: List[Path] = []

    # 1. 权重与卡片
    dst_weights = pkg / "best.pt"
    shutil.copy2(weights, dst_weights)
    files.append(dst_weights)

    run_dir = weights.parent.parent
    frame = collect_training(run_dir)
    metrics: Dict[str, float] = {}
    epochs = 0
    if frame is not None and not frame.empty:
        frame.columns = [c.strip() for c in frame.columns]
        epochs = len(frame)
        try:
            best_row = frame.loc[frame["metrics/mAP50-95(B)"].astype(float).idxmax()]
            for key in ("metrics/precision(B)", "metrics/recall(B)",
                        "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
                if key in frame.columns:
                    metrics[key] = round(float(best_row[key]), 4)
        except (KeyError, ValueError):
            metrics = {}

    card = {
        "model_file": dst_weights.name,
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_classes": len(project.class_names) if project else len(metrics) and None,
        "class_names": ({str(k): v for k, v in project.class_names.items()}
                        if project else {}),
        "dataset": (str(project.dataset_root) if project else ""),
        "source_run": run_dir.name,
        "trained_epochs": epochs,
        "best_metrics": metrics,
        "input_size": 640,
        "usage": "python predict.py --weights best.pt --data data.yaml --source <图片或视频>",
    }
    card_path = pkg / "model_card.json"
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(card_path)

    # 2. 数据配置
    dst_yaml = pkg / "data.yaml"
    if include_dataset and project:
        dataset_files, profile = _copy_dataset(Path(project.dataset_root), pkg)
        _write_portable_yaml(dst_yaml, profile)
        card["dataset"] = "dataset/"
        card["dataset_images"] = profile.total_images
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        files.append(dst_yaml)
        print(f"交付包已附带训练数据：{dataset_files} 个文件，{profile.total_images} 张图")
    elif project and Path(project.data_yaml).exists():
        shutil.copy2(project.data_yaml, dst_yaml)
        files.append(dst_yaml)

    # 3. 代码：入口脚本 + 源码包 + 任务脚本
    for name in ("train.py", "val.py", "predict.py", "check_dataset.py"):
        src = src_root / name
        if src.exists():
            shutil.copy2(src, pkg / name)
            files.append(pkg / name)

    for folder in ("lps", "scripts"):
        src = src_root / folder
        if not src.exists():
            continue
        dst = pkg / folder
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        files.append(dst)

    # 4. 依赖清单与预设方案配置
    req_src = src_root / "requirements.txt"
    if req_src.exists():
        shutil.copy2(req_src, pkg / "requirements.txt")
        files.append(pkg / "requirements.txt")

    # configs/ 里放着预设类别方案，lps 导入时会读；缺了虽然能降级运行，
    # 但用预设方案的脚本会少功能，所以一并带上
    cfg_src = src_root / "configs"
    if cfg_src.exists():
        dst_cfg = pkg / "configs"
        dst_cfg.mkdir(exist_ok=True)
        scheme = cfg_src / "class_scheme.yaml"
        if scheme.exists():
            shutil.copy2(scheme, dst_cfg / "class_scheme.yaml")
            files.append(dst_cfg / "class_scheme.yaml")

    # 5. 训练图表
    charts = pkg / "charts"
    charts.mkdir(exist_ok=True)
    for name in ("results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, charts / name)
            files.append(charts / name)

    # 6. 使用说明
    readme = _package_readme(stem, card)
    readme_path = pkg / "使用说明.md"
    readme_path.write_text(readme, encoding="utf-8")
    files.append(readme_path)

    return ExportResult(files=files)


def export_run_folder(run_dir: Path, out_parent: Path,
                      prefix: str = "train") -> Path:
    """把一次训练的完整产物导出成一个独立文件夹。

    内容与原始框架 `runs/detect/train` 完全一致：weights（best/last/分轮存档）、
    args.yaml（本次超参快照）、results.csv（逐轮指标）、results.png、混淆矩阵、
    PR/F1/P/R 曲线、训练批次图与验证对比图；另外附带本项目的 train.log。
    文件夹名带时间戳，多次训练不会互相覆盖。
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"训练结果目录不存在：{run_dir}")

    stamp = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y%m%d_%H%M")
    target = Path(out_parent) / f"{prefix}_{stamp}"
    if target.exists():
        # 同一分钟内导出两次时追加序号，避免互相覆盖
        index = 2
        while (Path(out_parent) / f"{prefix}_{stamp}_{index}").exists():
            index += 1
        target = Path(out_parent) / f"{prefix}_{stamp}_{index}"
    target.mkdir(parents=True, exist_ok=True)

    import shutil

    for item in sorted(run_dir.iterdir()):
        if item.is_dir():
            shutil.copytree(item, target / item.name, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target / item.name)
    return target


def _copy_dataset(dataset_root: Path, pkg: Path):
    """把数据集各划分的图像与标签带进包内 dataset/。

    优先硬链接（同分区不占额外空间），跨分区自动退化为复制，
    因此包可以整体拷贝到别的电脑，数据是自带的。
    """
    from ..data.profile import load_profile
    from ..data.splitter import link_or_copy

    profile = load_profile(Path(dataset_root))
    target_root = pkg / "dataset"
    count = 0
    for split, info in profile.splits.items():
        for source, kind in ((info.images_dir, "images"), (info.labels_dir, "labels")):
            if source is None:
                continue
            source = Path(source)
            if not source.exists():
                continue
            dst_dir = target_root / split / kind
            dst_dir.mkdir(parents=True, exist_ok=True)
            for file in source.iterdir():
                if file.is_file():
                    link_or_copy(file, dst_dir / file.name)
                    count += 1
    return count, profile


def _write_portable_yaml(path: Path, profile) -> None:
    """写一份不依赖绝对路径的 data.yaml，整个包搬走也能直接用。

    ultralytics 在 data.yaml 没有 ``path`` 键时，以该 yaml 自身所在目录为基准
    （见 data/utils.py：``data.get("path") or Path(data["yaml_file"]).parent``），
    所以这里省略 path，划分写成 dataset/<split>/images 即可随包移动。
    """
    import yaml as _yaml

    payload = {split: f"dataset/{split}/images" for split in profile.splits}
    payload["names"] = profile.names
    header = "# 由系统导出：路径相对本文件，整个包可整体移动\n"
    path.write_text(
        header + _yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _package_readme(stem: str, card: dict) -> str:
    """生成交付包内的使用说明。"""
    names = card.get("class_names") or {}
    class_lines = "\n".join(f"  {k}  {v}" for k, v in sorted(names.items(), key=lambda kv: int(kv[0])))
    metric_lines = "\n".join(f"- {k}：{v}" for k, v in (card.get("best_metrics") or {}).items())
    return f"""# 模型交付包：{stem}

本包可以直接在**装有 Python 的电脑**上用命令行使用，不需要安装原软件。

## 内容

| 文件 | 说明 |
| --- | --- |
| `best.pt` | 训练好的权重 |
| `model_card.json` | 类别表、类别数、来源实验、最优指标 |
| `data.yaml` | 本项目的数据集配置（路径为绝对路径） |
| `train.py` / `val.py` / `predict.py` / `check_dataset.py` | 训练、验证、推理、数据自检入口 |
| `lps/` | 源码包（含本项目自定义的 CBAM 注意力模块） |
| `scripts/` | 任务脚本 |
| `charts/` | 训练曲线与混淆矩阵 |
| `dataset/` | 训练数据（含 train/val/test 的图像与标签，导出时勾选才会包含） |
| `requirements.txt` | 依赖清单 |

## 类别（共 {card.get('num_classes') or len(names)} 类）

```
{class_lines}
```

## 最优指标

{metric_lines or '（未读到训练记录）'}

## 环境准备

```bash
cd {stem}_package
pip install -r requirements.txt
```

只做推理时，可安装 CPU 版 torch 以减小体积，参见
https://pytorch.org/get-started/locally/

## 用法

**验证模型**

```bash
python val.py --weights best.pt --data data.yaml --split val
```

**推理（图片 / 目录 / 视频 / 摄像头）**

```bash
python predict.py --weights best.pt --data data.yaml --source 图片.jpg
python predict.py --weights best.pt --data data.yaml --source 某个目录
python predict.py --weights best.pt --data data.yaml --source 视频.mp4
python predict.py --weights best.pt --data data.yaml --source 0 --max-frames 300
```

结果输出到 `outputs/predict/`，包含标注图、`detections.csv` 与 `summary.json`。

**重新训练**

```bash
python train.py --data data.yaml --epochs 100 --batch 8
```

注意：
- `data.yaml` 里的 `path` 指向原数据集目录，换机器后需要把该路径改成
  新机器上的实际位置，或把数据集一起拷过来后修改为相对路径。
- 训练需要与 `best.pt` 相同的类别体系，否则相当于重新训练一个新模型。
- 首次训练会自动查找 `yolov8n.pt` 预训练权重；没有则从零开始，精度会低一些。

## 数据自检

```bash
python check_dataset.py --data data.yaml --verify-images 20
```
"""

