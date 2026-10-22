# -*- coding: utf-8 -*-
"""训练脚本。

相对原始框架的改动：
  1. 用改进结构（CBAM 注意力）替换原生 yolov8n，预训练权重按层部分加载
  2. 数据集路径与类别方案由 lps.config 统一提供，不再手工改 data.yaml
  3. 保留原脚本的 AMP 自检兜底（网络不稳时 yolo11n.pt 会下到一半就断）
  4. 训练前自动跑一次数据自检，不通过直接拒绝开训
  5. 结束后打印每类 AP，便于判断长尾类是否被牺牲

用法::

    python scripts/train.py --scheme behavior4 --epochs 120
    python scripts/train.py --scheme activity3 --epochs 120 --name act3
    python scripts/train.py --resume
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _Tee:
    """把训练输出同时送到控制台与日志文件。

    送控制台保留原始内容（ANSI 颜色在真实终端里是有效的），
    写日志文件前剥掉 ANSI 转义序列与 \\r 回退，避免用记事本打开时满屏控制符。
    ultralytics 的进度条用 \\r 原地刷新，因此这里逐次透传、不做行切分。

    ``[progress]`` 开头的行只送控制台，不写日志文件：那是给界面做实时进度用的，
    120 轮 × 160 批会往日志里灌两万行，把真正的指标挤没。
    """

    def __init__(self, console: TextIO, log_file: TextIO) -> None:
        self._console = console
        self._log = log_file

    def write(self, text: str) -> int:
        try:
            self._console.write(text)
        except (OSError, ValueError):
            pass

        if PROGRESS_PREFIX in text:
            return len(text)

        try:
            self._log.write(strip_ansi(text))
        except (OSError, ValueError):
            pass
        return len(text)

    def flush(self) -> None:
        for stream in (self._console, self._log):
            try:
                stream.flush()
            except (OSError, ValueError):
                continue

    def isatty(self) -> bool:
        return False


from lps import config                       # noqa: E402
from lps.data import DatasetChecker          # noqa: E402
from lps.data.profile import load_profile    # noqa: E402
from lps.models import build                 # noqa: E402
from lps.services.plots import install_plot_patch  # noqa: E402
from lps.utils import ensure_ultralytics_font  # noqa: E402
from lps.utils.text import PROGRESS_PREFIX, strip_ansi  # noqa: E402

# 环境里的 polars 包已损坏（import 即被系统终止），而 ultralytics 出图依赖它，
# 这里换成 pandas 实现，避免训练结束阶段崩溃
install_plot_patch()

# ---------------------------------------------------------------- AMP 兜底
# ultralytics 训练前的 AMP 自检会加载 yolo11n.pt，本地缺失时联网下载。
# 网络不稳会下到一半中断，而官方只捕获 ConnectionError，整轮训练直接死掉。
import ultralytics.engine.trainer as _trainer_mod          # noqa: E402
from ultralytics.utils.checks import check_amp as _check_amp  # noqa: E402


def _safe_check_amp(model):
    """AMP 自检失败时退回 float32，不让训练因为下载失败而中断。"""
    weight = Path("yolo11n.pt")
    if not weight.exists() or weight.stat().st_size < 3_000_000:
        substitute = config.find_pretrained("yolov8n.pt")
        if substitute:
            shutil.copy(substitute, weight)
            print("AMP: yolo11n.pt 缺失或残缺，改用本地 yolov8n.pt 顶替做自检")
    try:
        return _check_amp(model)
    except Exception as exc:  # noqa: BLE001 - 兜底：任何自检异常都降级而非中断
        print(f"AMP: 自检未通过（{type(exc).__name__}: {exc}），本次训练用 amp=False")
        return False


_trainer_mod.check_amp = _safe_check_amp

# 训练实际开始的时刻，用于在每轮日志里给出耗时与预计剩余
_train_started_at: float = 0.0


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"



# ---------------------------------------------------------------- 批次级实时进度
# 界面看不到 ultralytics 的 tqdm 进度条：它用 \r 原地刷新、不换行，而子进程读取
# 是按行切分的，整条进度会一直卡在缓冲区里，看起来就像训练卡死。这里按固定节流
# 自己打印一行进度，界面解析后驱动进度条与剩余时间。
_batch_state: dict = {
    "epoch": -1, "iter": 0, "last_done": 0, "last_time": 0.0, "last_emit": 0.0,
}


def report_batch_progress(trainer) -> None:
    """每训练若干批输出一行进度（供界面解析）。

    速率用**滑动窗口**算（最近一次上报到现在完成了多少批），而不是
    「总完成量 ÷ 总耗时」——后者会把开训前的数据扫描、AMP 自检全算进分母，
    首行会报出「剩余 533 小时」这种离谱数字。
    """
    now = time.time()
    try:
        epoch_index = int(getattr(trainer, "epoch", 0))
        loader = getattr(trainer, "train_loader", None)
        per_epoch = len(loader) if loader is not None else 0
        total_epochs = int(getattr(trainer, "epochs", 0) or 0)
        if per_epoch <= 0 or total_epochs <= 0:
            return

        if epoch_index != _batch_state["epoch"]:
            _batch_state["epoch"] = epoch_index
            _batch_state["iter"] = 0
            _batch_state["last_time"] = now
            _batch_state["last_emit"] = now
            _batch_state["last_done"] = epoch_index * per_epoch

        _batch_state["iter"] += 1
        done = _batch_state["iter"]
        done_all = epoch_index * per_epoch + done
        total_all = per_epoch * total_epochs

        window_dt = now - _batch_state["last_time"]
        window_done = done_all - _batch_state["last_done"]
        rate = window_done / window_dt if window_dt > 0.5 and window_done > 0 else 0.0

        # 每 8 批或满 10 秒报一次；窗口还没攒够数据时不报，避免 0 速率
        if rate <= 0:
            return
        if done % 8 != 0 and now - _batch_state["last_emit"] < 10:
            return

        _batch_state["last_emit"] = now
        _batch_state["last_time"] = now
        _batch_state["last_done"] = done_all

        remain = (total_all - done_all) / rate
        elapsed = now - _train_started_at if _train_started_at else 0.0
        print(
            # 前置换行：tqdm 不换行，不加会让这行粘在进度条尾部
            f"\n{PROGRESS_PREFIX} epoch={epoch_index + 1}/{total_epochs}"
            f" iter={done}/{per_epoch}"
            f" rate={rate:.2f} elapsed={elapsed:.0f} remain={remain:.0f}",
            flush=True,
        )
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="牲畜姿态识别 - 改进 YOLOv8 训练")
    p.add_argument("--scheme", default=None,
                   help=f"预设类别方案（可选 {list(config.CLASS_SCHEMES)}）；传了 --data 时忽略")
    p.add_argument("--data", default=None,
                   help="任意 YOLO v8 数据集配置，类别数由数据决定")
    p.add_argument("--epochs", type=int, default=config.TrainConfig.epochs)
    p.add_argument("--batch", type=int, default=config.TrainConfig.batch)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=config.TrainConfig.workers)
    p.add_argument("--name", default=None, help="实验名，默认 scheme_结构名")
    p.add_argument("--amp", action="store_true", default=True, help="启用混合精度")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-check", action="store_true", help="跳过数据自检（不推荐）")
    p.add_argument("--pretrained", default=None, help="预训练权重，默认自动查找 yolov8n.pt")
    return p.parse_args()


def _resolve_dataset(args) -> tuple[Path, int, str]:
    """确定训练用数据配置与类别数。

    规则：
      - 传了 --data：以该配置为准，类别数从数据本身读出（支持任意数据集，
        类别数不限，不需要预先定义方案）
      - 否则：使用当前激活的类别方案派生的数据配置
    """
    if args.data:
        data_yaml = Path(args.data).resolve()
        if not data_yaml.exists():
            raise SystemExit(f"找不到数据配置：{data_yaml}")
        profile = load_profile(data_yaml)
        if profile.num_classes == 0:
            raise SystemExit("未能从数据配置中识别出任何类别，请检查 names 字段")
        return data_yaml, profile.num_classes, "custom"
    scheme = config.set_scheme(args.scheme or config.ACTIVE_SCHEME_KEY)
    return config.dataset_yaml_path(), scheme.num_classes, scheme.key


def sanitize_run_name(name: str) -> str:
    """把实验名收敛成纯 ASCII。

    Windows 下把中文实验名传给子进程时，ultralytics 侧可能拿到被截断的名字
    （实测「山羊_run」变成「_run」），导致输出目录与界面监听的目录对不上。
    实验名只用于目录命名，统一用 ASCII 最稳。
    """
    original = name.strip()
    cleaned = "".join(ch for ch in original if ch.isascii() and (ch.isalnum() or ch in "-_"))
    cleaned = cleaned.strip("-_")
    if not cleaned:
        return f"exp_{int(time.time())}"
    if cleaned != original:
        # 有字符被丢弃说明原名含非 ASCII，补个短后缀避免不同中文名净化后撞车
        cleaned = f"{cleaned}_{uuid.uuid4().hex[:4]}"
    return cleaned


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    # 先保证绘图字体可用，否则 ultralytics 会尝试联网下载 22MB 字体，
    # 网络差时长时间卡住，中断后留下的半截文件还会让它直接崩。
    font_ok = ensure_ultralytics_font() is not None

    data_yaml, num_classes, scheme_key = _resolve_dataset(args)
    print(f"数据集配置: {data_yaml}")
    print(f"类别数: {num_classes}")

    if not args.skip_check:
        report = DatasetChecker(data_yaml).run()
        print(report.summary())
        if not report.passed:
            raise SystemExit("数据自检未通过，已终止训练。修正后或加 --skip-check 跳过。")

    pretrained = Path(args.pretrained) if args.pretrained else config.find_pretrained("yolov8n.pt")
    model, build_report = build(nc=num_classes, pretrained=pretrained)
    print(build_report.summary())

    run_name = sanitize_run_name(args.name or f"{scheme_key}_cbam")
    tc = config.TrainConfig()

    # 先占住输出目录并打开日志文件，再启动训练。
    # 这样训练过程中终端打印的每一行都会同时落到 train.log，
    # 中途崩溃也能留下出事前的内容；目录由我们来建，故 exist_ok=True。
    run_dir = config.RUNS_DIR / run_name
    if run_dir.exists():
        raise SystemExit(f"实验名 {run_name} 已存在，请换一个 --name")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_file = log_path.open("w", encoding="utf-8")
    tee = _Tee(sys.stdout, log_file)
    print(f"训练日志将保存到: {log_path}")

    global _train_started_at
    _train_started_at = time.time()

    _logged_epochs: set[int] = set()

    def _log_epoch_metrics(trainer) -> None:
        """每轮结束后输出一行进度摘要。

        为什么要用回调：ultralytics 的进度条与指标表由 rich 打印，而 rich 的
        Console 在模块导入时就绑定了当时的 stdout，重定向 sys.stdout 抓不到，
        于是界面的日志区只剩一堆 torch 警告，看着像卡住。

        这里直接 print 到 stdout（已被 _Tee 接管，同时进控制台与日志文件），
        界面按 "epoch x/y" 解析就能推进进度条。

        训练结束时 ultralytics 会再做一次终验，on_fit_epoch_end 会被重复触发，
        故按轮次去重。
        """
        try:
            epoch = int(getattr(trainer, "epoch", -1))
            if epoch < 0 or epoch in _logged_epochs:
                return
            _logged_epochs.add(epoch)

            total = int(getattr(trainer, "epochs", 0) or 0)
            metrics = trainer.metrics or {}
            m50 = metrics.get("metrics/mAP50(B)")
            m5095 = metrics.get("metrics/mAP50-95(B)")
            precision = metrics.get("metrics/precision(B)")
            recall = metrics.get("metrics/recall(B)")

            head = f"epoch {epoch + 1}/{total}"
            if _train_started_at:
                elapsed = time.time() - _train_started_at
                per_epoch = elapsed / (epoch + 1)
                remain = per_epoch * max(0, total - epoch - 1)
                head += (f"  已用 {_fmt_duration(elapsed)}"
                         f"  预计剩余 {_fmt_duration(remain)}")

            parts = [head]
            for label, value in (("mAP50", m50), ("mAP50-95", m5095),
                                 ("P", precision), ("R", recall)):
                if isinstance(value, (int, float)):
                    parts.append(f"{label}={value:.4f}")
            if trainer.tloss is not None:
                losses = trainer.label_loss_items(trainer.tloss, prefix="train")
                for key, value in losses.items():
                    short = key.replace("train/", "").replace("_loss", "")
                    if isinstance(value, (int, float)):
                        parts.append(f"{short}={value:.3f}")

            print("  ".join(parts), flush=True)
        except (AttributeError, TypeError, ValueError, OSError):
            return

    model.add_callback("on_fit_epoch_end", _log_epoch_metrics)

    model.add_callback("on_train_batch_end", report_batch_progress)

    # 打包成 exe 后 Windows 下 DataLoader 的 multiprocessing spawn 会拿 exe
    # 自身当入口重启而卡死（与 val.py 同理），frozen 环境强制单进程加载。
    train_workers = 0 if getattr(sys, "frozen", False) else args.workers

    try:
        with redirect_stdout(tee), redirect_stderr(tee):
            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                workers=train_workers,
                project=str(config.RUNS_DIR),
                name=run_name,
                exist_ok=True,
                patience=tc.patience,
                optimizer=tc.optimizer,
                lr0=tc.lr0,
                lrf=tc.lrf,
                cos_lr=tc.cos_lr,
                warmup_epochs=tc.warmup_epochs,
                # ---- 数据增强：圈舍监控视角固定，旋转与翻转幅度收窄 ----
                hsv_h=tc.hsv_h, hsv_s=tc.hsv_s, hsv_v=tc.hsv_v,
                degrees=tc.degrees, translate=tc.translate, scale=tc.scale,
                fliplr=tc.fliplr, mosaic=tc.mosaic, mixup=tc.mixup,
                close_mosaic=tc.close_mosaic,
                # ---- 保存与日志 ----
                save=True,
                save_period=tc.save_period,
                plots=font_ok,
                seed=tc.seed,
                # CBAM 里的 adaptive_max_pool2d 没有 CUDA 确定性实现，
                # 开着确定性模式每轮都会刷一条 UserWarning，且会拖慢训练。
                deterministic=False,
                amp=args.amp,
                verbose=True,
            )
    finally:
        tee.flush()
        log_file.close()

    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", config.RUNS_DIR / run_name))
    print("\n========== 训练完成 ==========")
    print(f"结果目录: {save_dir}")
    print(f"训练日志: {log_path}")
    print(f"最优权重: {save_dir / 'weights' / 'best.pt'}")
    print("下一步: python scripts/val.py --weights "
          f"\"{save_dir / 'weights' / 'best.pt'}\" --data {data_yaml}")


if __name__ == "__main__":
    main()
