#!/usr/bin/env python
"""环境自检脚本（不训练，秒级完成）。

用途：换机器 / 重建环境 / 换数据源后，先跑它确认一切就绪，再开始训练。
校验项：
  1. 依赖包是否齐全且可导入
  2. datasets/ 下 12 个 txt 是否存在、列数是否为 26
  3. 每个子集的「测试引擎数 == RUL 行数」（防止静默错位）
  4. 特征选择是否按预期工作（各子集保留特征数）
  5. 四类模型能否正常构建（前向推理 1 个 batch）

用法：
    python tools/check_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# tools/ 供 `from config import ...`、项目根供 `from models.build import ...`
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    COL_NAMES,
    FD_SUBSETS,
    SETTING_NAMES,
    SENSOR_NAMES,
    WINDOW,
)
from preprocess import load_fd, select_features  # noqa: E402

OK, FAIL = "[OK]  ", "[FAIL]"
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{OK if condition else FAIL} {name}" + (f"  -> {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def check_imports() -> None:
    print("\n--- 1. 依赖包 ---")
    required = {
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
    }
    optional = {"onnx": "onnx", "onnxruntime": "onnxruntime", "flask": "flask"}
    for mod, pkg in required.items():
        try:
            __import__(mod)
            check(f"必需 {pkg}", True)
        except Exception as exc:  # noqa: BLE001
            check(f"必需 {pkg}", False, str(exc))
    for mod, pkg in optional.items():
        try:
            __import__(mod)
            check(f"可选 {pkg}（部署用）", True)
        except Exception:  # noqa: BLE001
            check(f"可选 {pkg}（部署用）", False, "未安装，仅影响 ONNX 导出/部署")


def check_data() -> None:
    print("\n--- 2. 数据文件与自洽性 ---")
    data_dir = ROOT / "datasets"
    if not data_dir.exists():
        check("datasets/ 目录存在", False, f"未找到 {data_dir}（见 SETUP.md 第 2 节）")
        return
    check("datasets/ 目录存在", True)

    for subset in FD_SUBSETS:
        # 文件齐全
        files = [data_dir / f"{kind}_{subset}.txt" for kind in ("train", "test", "RUL")]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            check(f"{subset} 三件套齐全", False, f"缺失 {missing}")
            continue
        check(f"{subset} 三件套齐全", True)

        # 列数
        train = load_fd(subset, data_dir)[0]
        n_cols = train.shape[1]
        check(
            f"{subset} 列数 == 26",
            n_cols == 26,
            f"实际 {n_cols}（unit+cycle+3 setting+21 sensor）",
        )

        # 引擎数 == RUL 行数（最关键：防止错位）
        _, _, rul = load_fd(subset, data_dir)
        test = np.loadtxt(data_dir / f"test_{subset}.txt")
        n_test_engines = int(np.unique(test[:, 0]).size)
        check(
            f"{subset} 测试引擎数 == RUL 行数",
            n_test_engines == rul.shape[0],
            f"test={n_test_engines}, RUL={rul.shape[0]}",
        )


def check_features() -> None:
    print("\n--- 3. 特征选择 ---")
    data_dir = ROOT / "datasets"
    if not (data_dir / "train_FD001.txt").exists():
        print("  (跳过：datasets/ 无数据)")
        return
    n_cand = len(SETTING_NAMES) + len(SENSOR_NAMES)
    print(f"  候选特征总数 = {n_cand}（3 setting + 21 sensor；unit/cycle 非特征）")
    for subset in FD_SUBSETS:
        try:
            train, _, _ = load_fd(subset, data_dir)
        except Exception as exc:  # noqa: BLE001
            check(f"{subset} 特征选择", False, str(exc))
            continue
        sel = select_features(train)
        check(
            f"{subset} 保留特征合理",
            0 < len(sel) <= n_cand,
            f"保留 {len(sel)}/{n_cand}（剔除 {n_cand - len(sel)} 个恒定特征）",
        )


def check_models() -> None:
    print("\n--- 4. 模型构建与前向推理 ---")
    try:
        import torch
        from models.build import build_model
    except Exception as exc:  # noqa: BLE001
        check("模型导入", False, str(exc))
        return

    n_feat = 18  # FD001 特征数，仅用于结构自检
    x = torch.randn(2, WINDOW, n_feat)
    for name in ("lstm", "bilstm", "cnn", "transformer"):
        try:
            model = build_model(name, n_feat)
            model.eval()
            with torch.no_grad():
                out = model(x)
            n_params = sum(p.numel() for p in model.parameters())
            check(
                f"{name} 前向通过",
                out.shape[0] == 2,
                f"输出 {tuple(out.shape)}, 参数量 {n_params:,}",
            )
        except Exception as exc:  # noqa: BLE001
            check(f"{name} 前向通过", False, str(exc))


def main() -> int:
    print("=" * 60)
    print("C-MAPSS RUL 项目 · 环境自检")
    print("=" * 60)
    check_imports()
    check_data()
    check_features()
    check_models()

    print("\n" + "=" * 60)
    if failures:
        print(f"自检未通过：{len(failures)} 项失败")
        for f in failures:
            print(f"  - {f}")
        print("=" * 60)
        return 1
    print("全部通过 ✅  可以开始训练了（见 README 第 7 节 / SETUP.md）")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
