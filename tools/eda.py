"""
时序数据探索性分析（EDA）—— 对应 JD「时序数据处理 / 特征工程」。

用原始 C-MAPSS 数据回答三个工程问题，并产出可直接进 README 的图与结论：
  1) 标签分布：测试集 RUL 直方图（看偏态 / 难度来源）
  2) 特征有效性：每个传感器在训练集的方差 → 解释 preprocess 为什么剔掉恒定传感器
  3) 退化信号：每个传感器与 RUL 的相关系数 → 说明哪些传感器真正携带退化信息
  4) 退化趋势：代表性传感器随「剩余寿命」变化的均值曲线 → 直观展示"临近失效如何漂移"

产物（docs/figures/eda/）：
  test_rul_distribution.png  sensor_variance.png  rul_correlation.png  degradation_trend.png
  eda_summary.json           （供 README 引用的结构化结论）

用法：python tools/eda.py --subset FD001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # EDA 脚本：屏蔽 groupby/cut 等无害告警，保持控制台干净

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from config import SETTING_NAMES, SENSOR_NAMES, VAR_THRESHOLD, FD_SUBSETS  # noqa: E402
from preprocess import load_fd, select_features                            # noqa: E402


def compute_rul_per_cycle(train_df: pd.DataFrame) -> pd.DataFrame:
    """训练集每个循环对应的 RUL = 该发动机最长循环 - 当前循环。"""
    df = train_df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (max_cycle - df["cycle"]).astype(float)
    return df


def fig_rul_distribution(rul_test: np.ndarray, out_path: Path, subset: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rul_test, bins=20, color="#576CBC", edgecolor="white")
    ax.set_title(f"{subset} test-set RUL distribution (n={len(rul_test)})")
    ax.set_xlabel("True RUL (cycles to failure)")
    ax.set_ylabel("Number of engines")
    ax.axvline(np.mean(rul_test), color="#00E0FF", lw=2, ls="--",
               label=f"mean={np.mean(rul_test):.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_sensor_variance(train_df: pd.DataFrame, out_path: Path, subset: str):
    var = train_df[SENSOR_NAMES].std()
    selected = set(select_features(train_df))
    colors = ["#00E0FF" if s in selected else "#E07A5F" for s in SENSOR_NAMES]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(1, 22), var.values, color=colors)
    ax.axhline(VAR_THRESHOLD, color="#888", ls="--", lw=1,
               label=f"drop threshold ({VAR_THRESHOLD:g})")
    ax.set_xticks(range(1, 22))
    ax.set_xlabel("Sensor index")
    ax.set_ylabel("Std (training set)")
    ax.set_title(f"{subset} sensor variance — cyan=kept, orange=dropped (near-constant)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def correlation_with_rul(train_df: pd.DataFrame) -> pd.Series:
    """逐列计算与 RUL 的 Pearson 相关系数；恒定特征（std≈0）标为 NaN。"""
    cols = SETTING_NAMES + SENSOR_NAMES
    t = train_df["rul"].values
    out = {}
    for c in cols:
        x = train_df[c].values
        if np.std(x) < 1e-9:
            out[c] = np.nan
            continue
        out[c] = float(np.corrcoef(x, t)[0, 1])
    return pd.Series(out).abs()


def fig_rul_correlation(corr: pd.Series, out_path: Path, subset: str,
                        selected: list[str]):
    corr_plot = corr.fillna(0).sort_values(ascending=True)
    colors = ["#00E0FF" if c in selected else "#E07A5F" for c in corr_plot.index]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.barh(range(len(corr_plot)), corr_plot.values, color=colors)
    ax.set_yticks(range(len(corr_plot)))
    ax.set_yticklabels(corr_plot.index, fontsize=8)
    ax.set_xlabel("|Pearson corr| with RUL")
    ax.set_title(f"{subset} feature relevance to RUL — cyan=kept, orange=dropped")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_degradation_trend(train_df: pd.DataFrame, top_sensors: list[str],
                          out_path: Path, subset: str):
    max_rul = train_df["rul"].max()
    bins = np.linspace(0, max_rul, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(top_sensors):
        # 按 RUL 分箱求均值，跨发动机对齐"剩余寿命"
        grp = train_df.groupby(pd.cut(train_df["rul"], bins, include_lowest=True))[s].mean()
        grp = grp.reindex(range(len(centers)))
        ax.plot(centers, grp.values, marker="o", ms=3, label=s, color=cmap(i % 10))
    ax.set_xlabel("Remaining useful life RUL (cycles) — left = near failure")
    ax.set_ylabel("Mean sensor value (raw)")
    ax.set_title(f"{subset} degradation trend of top-RUL-correlated sensors")
    ax.invert_xaxis()  # 让"临近失效"在左侧，退化趋势向右看更直观
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="EDA for C-MAPSS RUL")
    p.add_argument("--subset", default="FD001", choices=FD_SUBSETS)
    p.add_argument("--data_dir", default=str(ROOT / "datasets"))
    p.add_argument("--out_dir", default=str(ROOT / "docs" / "figures" / "eda"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, _, rul_test = load_fd(args.subset, args.data_dir)
    train_df = compute_rul_per_cycle(train_df)

    selected = select_features(train_df)
    dropped = [s for s in SENSOR_NAMES if s not in selected]

    print(f"[{args.subset}] engines(train)={train_df['unit'].nunique()}  "
          f"cycles={len(train_df)}  test_engines={len(rul_test)}")
    print(f"  selected features ({len(selected)}): {selected}")
    print(f"  dropped sensors ({len(dropped)}): {dropped}")

    # 1) 标签分布
    fig_rul_distribution(rul_test, out_dir / "test_rul_distribution.png", args.subset)
    # 2) 特征有效性（方差）
    fig_sensor_variance(train_df, out_dir / "sensor_variance.png", args.subset)
    # 3) RUL 相关性
    corr = correlation_with_rul(train_df)
    fig_rul_correlation(corr, out_dir / "rul_correlation.png", args.subset, selected)
    ranked = corr.dropna().sort_values(ascending=False)
    print("  top-6 RUL-correlated features:")
    for name, val in ranked.head(6).items():
        print(f"    {name:<10} |corr|={val:.3f}")
    # 4) 退化趋势（取相关性最高的若干「传感器」，跳过恒定/工况列）
    top_sensors = [c for c in ranked.index if c in SENSOR_NAMES][:6]
    fig_degradation_trend(train_df, top_sensors, out_dir / "degradation_trend.png", args.subset)

    summary = {
        "subset": args.subset,
        "n_train_engines": int(train_df["unit"].nunique()),
        "n_train_cycles": int(len(train_df)),
        "n_test_engines": int(len(rul_test)),
        "test_rul_mean": float(np.mean(rul_test)),
        "test_rul_std": float(np.std(rul_test)),
        "selected_features": selected,
        "dropped_sensors": dropped,
        "top_rul_correlated": {k: float(v) for k, v in ranked.head(8).items()},
    }
    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nEDA 图已保存至 {out_dir}")
    print(f"摘要: {out_dir / 'eda_summary.json'}")


if __name__ == "__main__":
    main()
