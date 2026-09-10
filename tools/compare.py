"""
汇总各模型在指定子集上的测试结果，生成对比表与对比柱状图。

读取 runs/<model>_<subset>/metrics.json，筛选子集后按 RMSE 排序，
输出：docs/figures/model_compare.png（RMSE/Score 双轴柱状图）+ docs/figures/compare_<subset>.json。

用法：python tools/compare.py --subset FD001
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from metrics import plot_compare_bar  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="FD001")
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--out_dir", default=str(ROOT / "docs" / "figures"))
    return p.parse_args()


def main():
    args = parse_args()
    rows = []
    for p in sorted(glob.glob(f"{args.runs}/*/metrics.json")):
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("subset") == args.subset:
            rows.append(m)
    if not rows:
        print(f"未找到 {args.subset} 的 metrics.json，请先训练。")
        return

    rows.sort(key=lambda r: r["test_rmse"])
    models = [r["model"] for r in rows]
    rmses = [r["test_rmse"] for r in rows]
    scores = [r["test_score"] for r in rows]

    print(f"\n{'模型':<14}{'测试 RMSE':>12}{'测试 Score':>14}")
    print("-" * 42)
    for r in rows:
        print(f"{r['model']:<14}{r['test_rmse']:>12.3f}{r['test_score']:>14.1f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_compare_bar(models, rmses, scores, str(out_dir / "model_compare.png"), args.subset)
    with open(out_dir / f"compare_{args.subset}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n对比图：{out_dir / 'model_compare.png'}")
    print(f"对比数据：{out_dir / f'compare_{args.subset}.json'}")


if __name__ == "__main__":
    main()
