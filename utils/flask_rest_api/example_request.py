"""
Flask 推理服务示例客户端（无需额外依赖，使用标准库 urllib）。

用法：
  1) 终端 A：python utils/flask_rest_api/restapi.py --model lstm --subset FD001
  2) 终端 B：python utils/flask_rest_api/example_request.py --model lstm --subset FD001

本脚本从真实测试集中取某台发动机最近若干循环（原始 26 列），
组装请求发给 /predict，并打印「预测 RUL vs 真实 RUL」。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lstm")
    p.add_argument("--subset", default="FD001")
    p.add_argument("--unit", type=int, default=1, help="测试集中的发动机编号")
    p.add_argument("--url", default="http://127.0.0.1:5000")
    p.add_argument("--data_dir", default=str(ROOT / "datasets"))
    return p.parse_args()


def main():
    args = parse_args()
    raw = np_loadtxt(Path(args.data_dir) / f"test_{args.subset}.txt")
    mask = raw[:, 0] == args.unit
    cycles = raw[mask][:, :26]  # 取该发动机全部循环（服务会自动取最后 window 个）

    # 真实 RUL（官方给出，对应测试集每条轨迹末尾）
    rul = np_loadtxt(Path(args.data_dir) / f"RUL_{args.subset}.txt")
    true_rul = float(rul[args.unit - 1])

    body = {"engine_id": f"{args.subset}_unit{args.unit:03d}", "cycles": cycles.tolist()}
    req = urllib.request.Request(
        args.url + "/predict",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    print(f"发动机 {args.unit}（{args.subset}）")
    print(f"  预测 RUL : {result.get('rul')} cycles")
    print(f"  真实 RUL : {true_rul:.1f} cycles")
    print(f"  绝对误差 : {abs(result.get('rul', 0) - true_rul):.2f} cycles")


def np_loadtxt(path: Path):
    import numpy as np
    return np.loadtxt(str(path))


if __name__ == "__main__":
    main()
