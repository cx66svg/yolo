# -*- coding: utf-8 -*-
"""数据集自检入口。

原始框架的 README 提到了 check_dataset.py，但压缩包里并没有这个文件。
这里补上，并把检查能力接到 lps.data.checker 上。

用法::

    python check_dataset.py                        # 检查当前方案的数据配置
    python check_dataset.py --data configs/data_pose_view8.yaml
    python check_dataset.py --verify-images 20     # 额外抽查 20 张图能否解码
    python check_dataset.py --duplicates           # 跨划分查重复图像
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lps import config  # noqa: E402
from lps.data.checker import (  # noqa: E402
    IMAGE_EXTENSIONS, DatasetChecker, find_duplicates, quick_verify_images,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO 数据集自检")
    p.add_argument("--data", default=None, help="数据配置，默认用当前类别方案")
    p.add_argument("--verify-images", type=int, default=0,
                   help="每个划分抽查多少张图能否解码，0 表示不抽查")
    p.add_argument("--duplicates", action="store_true", help="跨划分查找重复图像")
    p.add_argument("--max-issues", type=int, default=20, help="每类问题最多打印多少条")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data) if args.data else config.dataset_yaml_path()
    if not data_yaml.exists():
        raise SystemExit(f"找不到数据配置：{data_yaml}\n请先运行 python scripts/prepare_data.py")

    report = DatasetChecker(data_yaml, max_issues_per_type=args.max_issues).run()
    print(report.summary())

    for issue in report.issues:
        print(f"  [{issue.level}] {issue.path}  {issue.message}")

    if args.verify_images:
        print("\n抽查图像可解码性…")
        root = data_yaml.parent
        dirs = []
        for split in ("train", "valid", "test"):
            d = config.DATA_ROOT / split / "images"
            if d.exists():
                dirs.append(d)
        problems = quick_verify_images(dirs, sample=args.verify_images)
        if problems:
            for p in problems[: args.max_issues]:
                print(f"  [error] {p.path}  {p.message}")
        else:
            print(f"  抽查 {args.verify_images} 张/划分，全部可正常解码")

    if args.duplicates:
        print("\n跨划分查重…")
        dirs = [config.DATA_ROOT / s / "images" for s in ("train", "valid", "test")]
        dirs = [d for d in dirs if d.exists()]
        dups = find_duplicates(dirs)
        if dups:
            print(f"  发现 {len(dups)} 组重复图像：")
            for _, paths in list(dups.items())[: args.max_issues]:
                print("   ", " | ".join(Path(p).name for p in paths))
        else:
            print("  未发现重复图像")

    print("\n结论：", "通过" if report.passed else "未通过，请先修正错误")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
