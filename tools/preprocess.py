"""
C-MAPSS 数据预处理管线。

流程：
  1) 解析 train/test/RUL 三个 txt（26 列空格分隔）
  2) 特征选择：剔除方差接近 0 的恒定传感器（按训练集统计），保留 3 个工况设置
  3) 归一化：z-score，仅用训练集统计量拟合，再变换测试集（防止信息泄漏）
  4) 构造滑动窗口样本：训练集对每条发动机轨迹滑窗增广；测试集取每条轨迹最后 W 个循环
  5) RUL 标签：训练集 RUL(t)=轨迹最长循环 - t，并裁剪到 [0, CLIP_RUL]；
              测试集 RUL 直接取官方给出的 RUL_FD00X.txt

输出 prepare_fd() 返回 dict，含训练/验证/测试数组、所用特征、归一化统计量，
供训练、评测、ONNX 导出与推理服务复用（必须保证预处理一致）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    COL_NAMES, SETTING_NAMES, SENSOR_NAMES,
    WINDOW, CLIP_RUL, STRIDE, VAL_RATIO, VAR_THRESHOLD,
)


def load_raw(path: str | Path) -> pd.DataFrame:
    """读取单个 txt（无表头、空格分隔），列名按 C-MAPSS 规范。"""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)


def load_fd(subset: str, data_dir: str | Path):
    """加载某子集的 train / test / RUL。返回 (train_df, test_df, rul_array)。"""
    d = Path(data_dir)
    train = load_raw(d / f"train_{subset}.txt")
    test = load_raw(d / f"test_{subset}.txt")
    rul = pd.read_csv(d / f"RUL_{subset}.txt", sep=r"\s+", header=None,
                      names=["rul"]).values.flatten().astype(float)
    return train, test, rul


def select_features(train_df: pd.DataFrame, threshold: float = VAR_THRESHOLD) -> list[str]:
    """按训练集统计选择特征：保留 3 个工况设置 + 方差 > threshold 的传感器。"""
    selected_sensors = [s for s in SENSOR_NAMES if train_df[s].std() > threshold]
    return SETTING_NAMES + selected_sensors


def _split_engines(units: np.ndarray, val_ratio: float, rng: np.random.Generator):
    """把发动机编号随机切成 训练/验证 两组（按 engine 切分，避免同机泄漏）。"""
    uniq = np.unique(units)
    n_val = max(1, int(round(len(uniq) * val_ratio)))
    idx = rng.permutation(len(uniq))[:n_val]
    val_units = set(uniq[idx].tolist())
    return val_units


def _to_windows(seq: np.ndarray, window: int, stride: int):
    """对单条轨迹( T x F )滑窗，返回窗口数组 (N x window x F)。"""
    T = seq.shape[0]
    if T < window:
        # 长度不足时向前复制首行补齐
        pad = np.repeat(seq[:1], window - T, axis=0)
        seq = np.concatenate([pad, seq], axis=0)
        T = seq.shape[0]
    xs = [seq[i:i + window] for i in range(0, T - window + 1, stride)]
    return np.array(xs)


def _rul_labels(seq_len: int, clip: int):
    """训练集 RUL 标签：轨迹最长循环处为 0，向前线性递增，裁剪到 [0, clip]。"""
    rul = (np.arange(seq_len - 1, -1, -1))  # cycle1..T -> T-1 .. 0
    return np.clip(rul, 0, clip).astype(float)


def prepare_fd(subset: str, data_dir: str | Path,
               window: int = WINDOW, clip_rul: int = CLIP_RUL,
               stride: int = STRIDE, val_ratio: float = VAL_RATIO,
               seed: int = 42) -> dict:
    """构造某子集的完整训练/验证/测试样本。返回 dict（含所有复用所需元信息）。"""
    train_df, test_df, rul_test = load_fd(subset, data_dir)

    feature_cols = select_features(train_df)
    n_features = len(feature_cols)

    # 仅用训练集拟合归一化统计量（防泄漏）
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0, 1.0)

    def normalize(df: pd.DataFrame) -> np.ndarray:
        return ((df[feature_cols] - mean) / std).values.astype(np.float32)

    train_x = normalize(train_df)
    test_x = normalize(test_df)

    rng = np.random.default_rng(seed)
    val_units = _split_engines(train_df["unit"].values, val_ratio, rng)

    X_train, y_train, X_val, y_val = [], [], [], []
    # 训练集：按 engine 分组后滑窗；并按 engine 切分训练/验证
    for unit, grp in train_df.groupby("unit"):
        seq = train_x[grp.index.to_numpy()]  # 与 train_df 同序（默认 RangeIndex → 位置即标签）
        T = seq.shape[0]
        rul = _rul_labels(T, clip_rul)
        wins = _to_windows(seq, window, stride)
        end_idx = np.arange(window - 1, T, stride)[: len(wins)]
        labels = rul[end_idx]
        if unit in val_units:
            X_val.append(wins)
            y_val.append(labels)
        else:
            X_train.append(wins)
            y_train.append(labels)

    # 测试集：每条轨迹取最后 window 个循环作为单个样本
    X_test, y_test = [], []
    for j, (unit, grp) in enumerate(test_df.groupby("unit")):
        seq = test_x[grp.index.to_numpy()]
        win = _to_windows(seq, window, stride=1)  # 返回 (>=1, window, F)
        X_test.append(win[-1])  # 最后一段窗口
        y_test.append(float(rul_test[j]))

    out = dict(
        subset=subset,
        feature_cols=feature_cols,
        n_features=n_features,
        window=window,
        clip_rul=clip_rul,
        mean=mean.to_dict(),
        std=std.to_dict(),
        X_train=np.concatenate(X_train).astype(np.float32),
        y_train=np.concatenate(y_train).astype(np.float32),
        X_val=np.concatenate(X_val).astype(np.float32),
        y_val=np.concatenate(y_val).astype(np.float32),
        X_test=np.array(X_test, dtype=np.float32),
        y_test=np.array(y_test, dtype=np.float32),
    )
    return out


def summarize(data: dict) -> str:
    """返回一份可读的数据概览，便于日志/README 引用。"""
    return (
        f"[{data['subset']}] features={data['n_features']} window={data['window']} clip={data['clip_rul']}\n"
        f"  train={data['X_train'].shape[0]}  val={data['X_val'].shape[0]}  "
        f"test={data['X_test'].shape[0]}  X.shape={tuple(data['X_train'].shape[1:])}"
    )


if __name__ == "__main__":
    import json
    from config import FD_SUBSETS

    base = Path(__file__).resolve().parents[1] / "datasets"
    for sub in FD_SUBSETS:
        d = prepare_fd(sub, base)
        print(summarize(d))
        print("  feature_cols:", d["feature_cols"])
