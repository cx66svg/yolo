# -*- coding: utf-8 -*-
"""数据准备：按类别方案派生数据集，并输出统计与属性表。

用法::

    python scripts/prepare_data.py                 # 处理当前激活方案
    python scripts/prepare_data.py --all           # 全部方案都生成
    python scripts/prepare_data.py --scheme activity3 --stats
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lps import config                                  # noqa: E402
from lps.data import DatasetChecker, DatasetConverter, DatasetStats  # noqa: E402
from lps.data import attributes as attr                  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="派生数据集并统计")
    p.add_argument("--scheme", default=config.ACTIVE_SCHEME_KEY)
    p.add_argument("--all", action="store_true", help="生成全部方案的派生数据集")
    p.add_argument("--stats", action="store_true", help="额外输出统计与属性表")
    p.add_argument("--source", default=None, help="原始数据集根目录，默认取 config.DATA_ROOT")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    converter = DatasetConverter(args.source)
    keys = list(config.CLASS_SCHEMES) if args.all else [args.scheme]

    for key in keys:
        scheme = config.CLASS_SCHEMES[key]
        result = converter.convert(scheme)
        print(result.summary())

        report = DatasetChecker(result.yaml_path).run()
        print(report.summary())
        if not report.passed:
            print("  警告：该方案数据自检未通过，请先修正")
        print()

    if args.stats:
        stats = DatasetStats.compute(config.DATA_ROOT)
        print("=== 原始数据统计 ===")
        print(stats.summary())
        stats.export_csv(config.OUTPUT_DIR / "attributes" / "dataset_stats.csv")

        records = attr.derive_all(config.DATA_ROOT)
        attr.export_csv(records, config.OUTPUT_DIR / "attributes" / "object_attributes.csv")
        print("\n=== 目标属性分布 ===")
        print(attr.summarize(records, DatasetStats.compute(config.DATA_ROOT).class_names))
        print(f"\n属性表: {config.OUTPUT_DIR / 'attributes' / 'object_attributes.csv'}")


if __name__ == "__main__":
    main()
