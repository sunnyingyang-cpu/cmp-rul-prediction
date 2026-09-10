"""
评估指标与绘图工具。

- rmse：均方根误差（C-MAPSS 主指标之一）。
- score：NASA 竞赛非对称得分——晚预测（高估剩余寿命）惩罚更重，
         因设备在预测寿命耗尽后仍运行会带来更大安全风险。
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def score(y_true: np.ndarray, y_pred: np.ndarray, a: float = 13.0, b: float = 10.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / a) - 1.0, np.exp(d / b) - 1.0)
    return float(np.sum(s))


def plot_history(train_loss: list[float], val_loss: list[float], path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="train MSE")
    plt.plot(val_loss, label="val MSE")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("Training / Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def plot_pred_true(y_true: np.ndarray, y_pred: np.ndarray, path: str,
                   title: str = "Predicted vs True RUL") -> None:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    order = np.argsort(y_true)
    plt.figure(figsize=(6, 4))
    plt.plot(y_true[order], y_true[order], "k--", label="ideal")
    plt.plot(y_true[order], y_pred[order], "b-", label="predicted")
    plt.xlabel("True RUL (cycles)"); plt.ylabel("Predicted RUL")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def plot_compare_bar(models: list[str], rmse_vals: list[float], score_vals: list[float],
                     path: str, subset: str = "") -> None:
    """多模型 RMSE / Score 对比柱状图（双轴）。"""
    x = np.arange(len(models))
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    bars = ax1.bar(x - 0.2, rmse_vals, width=0.4, color="#576CBC", label="RMSE")
    ax1.set_ylabel("RMSE (cycles)", color="#576CBC")
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=15)
    for b, v in zip(bars, rmse_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + 0.2, score_vals, width=0.4, color="#00A6A6", label="Score")
    ax2.set_ylabel("Score (lower better)", color="#00A6A6")
    for b, v in zip(bars2, score_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax1.set_title(f"Model Comparison {subset}".strip())
    fig.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
