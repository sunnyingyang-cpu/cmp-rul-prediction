"""
传统机器学习基线（对应 JD：特征工程 + 回归建模）。

做法：复用与深度学习完全相同的滑动窗口输入（每条样本 = 最近 W 个循环 × F 个特征），
展平为 W×F 维特征向量，训练 HistGradientBoostingRegressor 回归 RUL。
这样与 DL 路线输入完全一致，对比才公平，也体现"传统方法也能做、且可作为基线/对照"的工程认知。

输出（runs/baseline_FD001/）：metrics.json + pred_vs_true.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from config import SCORE_A, SCORE_B, WINDOW, CLIP_RUL, TRAIN_DEFAULTS  # noqa: E402
from preprocess import prepare_fd, summarize                           # noqa: E402
from metrics import rmse, score, plot_pred_true                        # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Traditional ML baseline for RUL")
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--data_dir", default=str(ROOT / "datasets"))
    p.add_argument("--out", default=str(ROOT / "runs"))
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--clip", type=int, default=CLIP_RUL)
    p.add_argument("--seed", type=int, default=TRAIN_DEFAULTS["seed"])
    return p.parse_args()


def main():
    args = parse_args()
    data = prepare_fd(args.subset, args.data_dir, window=args.window,
                      clip_rul=args.clip, seed=args.seed)
    print(summarize(data))

    N, W, F = data["X_train"].shape
    X_train = data["X_train"].reshape(N, W * F)
    X_test = data["X_test"].reshape(data["X_test"].shape[0], W * F)
    y_train = data["y_train"]
    y_test = data["y_test"]

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=None,
        l2_regularization=1.0, random_state=args.seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_rmse = rmse(y_test, y_pred)
    test_score = score(y_test, y_pred, SCORE_A, SCORE_B)

    out_dir = Path(args.out) / f"baseline_{args.subset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "hist_gbrt",
        "subset": args.subset,
        "n_features_in": W * F,
        "test_rmse": test_rmse,
        "test_score": test_score,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    plot_pred_true(y_test, y_pred, str(out_dir / "pred_vs_true.png"),
                   title=f"baseline {args.subset} · Pred vs True RUL")

    print(f"\n=== baseline {args.subset} 测试集 ===")
    print(f"  RMSE = {test_rmse:.3f}  cycles")
    print(f"  Score= {test_score:.1f}")
    print(f"  结果已保存至 {out_dir}")


if __name__ == "__main__":
    main()
