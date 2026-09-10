"""
设备剩余寿命（RUL）在线推理服务。

加载 ONNX 模型（tools/export_onnx.py 导出）与预处理元信息，复现训练时的
特征选择与 z-score 归一化，对最近 window 个循环预测 RUL。

请求示例（POST /predict）：
  {
    "engine_id": "FD001_001",
    "cycles": [ [1, 1, 0.1, 0.2, ...(26 列，单位/循环/3设置/21传感器)...], ... ]  // 最近若干循环
  }
响应：
  { "engine_id": "...", "rul": 78.3, "unit": "cycles" }

启动：
  pip install flask onnxruntime
  python utils/flask_rest_api/restapi.py --model lstm --subset FD001 --port 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from flask import Flask, request, jsonify

ROOT = Path(__file__).resolve().parents[2]
# C-MAPSS 26 列顺序：unit, cycle, setting1-3, sensor1-21
COL_ORDER = (["unit", "cycle"] + [f"setting{i}" for i in range(1, 4)]
             + [f"sensor{i}" for i in range(1, 22)])


def load_artifacts(model: str, subset: str, weights_dir: Path):
    meta_path = weights_dir / f"{model}_{subset}_preprocess.json"
    onnx_path = weights_dir / f"{model}_{subset}.onnx"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    feature_idx = [COL_ORDER.index(c) for c in meta["feature_cols"]]
    mean = np.array([meta["mean"][c] for c in meta["feature_cols"]], dtype=np.float32)
    std = np.array([meta["std"][c] for c in meta["feature_cols"]], dtype=np.float32)
    std[std == 0] = 1.0
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return meta, feature_idx, mean, std, session


def preprocess(cycles: list, feature_idx, mean, std, window: int) -> np.ndarray:
    arr = np.array(cycles, dtype=np.float32)          # (T, 26)
    if arr.ndim != 2 or arr.shape[1] != 26:
        raise ValueError("每个 cycle 必须是 26 个数值（unit,cycle,3设置,21传感器）")
    feat = arr[:, feature_idx]                          # (T, F)
    if feat.shape[0] < window:
        pad = np.repeat(feat[:1], window - feat.shape[0], axis=0)
        feat = np.concatenate([pad, feat], axis=0)
    windowed = feat[-window:]                           # 取最近 window 个循环
    norm = (windowed - mean) / std
    return norm[None, ...]                              # (1, window, F)


def create_app(model: str, subset: str, weights_dir: Path):
    if ort is None:
        raise ImportError("请先安装 onnxruntime：pip install onnxruntime")
    meta, feature_idx, mean, std, session = load_artifacts(model, subset, weights_dir)
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model": model, "subset": subset,
                        "window": meta["window"], "n_features": meta["n_features"]})

    @app.post("/predict")
    def predict():
        body = request.get_json(force=True)
        if not body or "cycles" not in body:
            return jsonify({"error": "缺少 cycles 字段"}), 400
        try:
            x = preprocess(body["cycles"], feature_idx, mean, std, meta["window"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        out = session.run(None, {"input": x})[0]
        rul = float(np.clip(out[0, 0], 0, None))
        return jsonify({
            "engine_id": body.get("engine_id", ""),
            "rul": round(rul, 2),
            "unit": "cycles",
        })

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lstm")
    p.add_argument("--subset", default="FD001")
    p.add_argument("--weights_dir", default=str(ROOT / "weights"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app = create_app(args.model, args.subset, Path(args.weights_dir))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
