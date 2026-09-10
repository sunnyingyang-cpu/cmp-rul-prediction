"""
将训练得到的最优模型导出为 ONNX（跨平台推理，可脱离 PyTorch 部署）。

用法：
  python tools/export_onnx.py --model lstm --subset FD001

产物：
  weights/<model>_<subset>.onnx            模型（输入 batch×window×F，输出 batch×1）
  weights/<model>_<subset>_preprocess.json 预处理元信息（特征列/均值/标准差/窗口），供推理端复现
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from models.build import build_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Export RUL model to ONNX")
    p.add_argument("--model", default="lstm", choices=["lstm", "bilstm", "cnn", "transformer"])
    p.add_argument("--subset", default="FD001")
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--out", default=str(ROOT / "weights"))
    p.add_argument("--batch", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.runs) / f"{args.model}_{args.subset}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = build_model(ckpt["model_name"], ckpt["n_features"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{args.model}_{args.subset}.onnx"

    dummy = torch.randn(args.batch, ckpt["window"], ckpt["n_features"])
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["rul"],
        dynamic_axes={"input": {0: "batch"}, "rul": {0: "batch"}},
        opset_version=13,
    )

    meta = {
        "model_name": ckpt["model_name"],
        "feature_cols": ckpt["feature_cols"],
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "window": ckpt["window"],
        "clip_rul": ckpt["clip_rul"],
        "n_features": ckpt["n_features"],
    }
    meta_path = out_dir / f"{args.model}_{args.subset}_preprocess.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"导出完成：{onnx_path}")
    print(f"预处理元信息：{meta_path}")
    print(f"  输入: ({args.batch}, {ckpt['window']}, {ckpt['n_features']})  输出: ({args.batch}, 1)")


if __name__ == "__main__":
    main()
