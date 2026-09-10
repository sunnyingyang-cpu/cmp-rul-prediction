"""
训练脚本：在指定 FD 子集上训练某一路线模型，保存最优检查点并评测测试集。

用法示例：
  python tools/train.py --model lstm --subset FD001 --device cpu
  python tools/train.py --model transformer --subset FD001 --epochs 60

输出（runs/<model>_<subset>/）：
  best.pt         最优模型权重 + 预处理元信息（供评测/导出/部署复用）
  metrics.json    测试集 RMSE / Score / 参数量等
  history.png     训练/验证损失曲线
  pred_vs_true.png 测试集预测 vs 真实 RUL
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from config import MODEL_DEFAULTS, TRAIN_DEFAULTS, SCORE_A, SCORE_B  # noqa: E402
from preprocess import prepare_fd, summarize                           # noqa: E402
from metrics import rmse, score, plot_history, plot_pred_true         # noqa: E402
from models.build import build_model                                   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train RUL model on C-MAPSS")
    p.add_argument("--model", default="lstm", choices=["lstm", "bilstm", "cnn", "transformer"])
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--data_dir", default=str(ROOT / "datasets"))
    p.add_argument("--out", default=str(ROOT / "runs"))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--clip", type=int, default=125)
    p.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS["epochs"])
    p.add_argument("--batch_size", type=int, default=TRAIN_DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=TRAIN_DEFAULTS["lr"])
    p.add_argument("--weight_decay", type=float, default=TRAIN_DEFAULTS["weight_decay"])
    p.add_argument("--patience", type=int, default=TRAIN_DEFAULTS["patience"])
    p.add_argument("--seed", type=int, default=TRAIN_DEFAULTS["seed"])
    return p.parse_args()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = prepare_fd(args.subset, args.data_dir, window=args.window,
                      clip_rul=args.clip, seed=args.seed)
    print(summarize(data))

    device = torch.device("cuda" if (torch.cuda.is_available() and args.device == "cuda") else "cpu")
    print(f"device={device}")

    model_kwargs = MODEL_DEFAULTS.get(args.model, {})
    model = build_model(args.model, data["n_features"], **model_kwargs).to(device)
    n_params = count_params(model)
    print(f"model={args.model}  params={n_params}")

    train_ds = TensorDataset(torch.from_numpy(data["X_train"]), torch.from_numpy(data["y_train"]))
    val_ds = TensorDataset(torch.from_numpy(data["X_val"]), torch.from_numpy(data["y_val"]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out) / f"{args.model}_{args.subset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mse = float("inf")
    patience = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_DEFAULTS["clip_grad"])
            optimizer.step()
            running += loss.item() * len(xb)
        train_mse = running / len(train_ds)

        model.eval()
        vrunning = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                vrunning += criterion(pred, yb).item() * len(xb)
        val_mse = vrunning / len(val_ds)
        history.append((train_mse, val_mse))

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience = 0
            torch.save({
                "model_name": args.model,
                "model_kwargs": model_kwargs,
                "n_features": data["n_features"],
                "window": data["window"],
                "clip_rul": data["clip_rul"],
                "feature_cols": data["feature_cols"],
                "mean": data["mean"],
                "std": data["std"],
                "model_state": model.state_dict(),
            }, out_dir / "best.pt")
        else:
            patience += 1

        print(f"epoch {epoch+1:3d}/{args.epochs}  train_mse={train_mse:.4f}  "
              f"val_mse={val_mse:.4f}  val_rmse={val_mse**0.5:.3f}  patience={patience}")
        if patience >= args.patience:
            print("early stopping.")
            break

    # ---- 评测测试集（加载最优权重） ----
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    X_test = torch.from_numpy(data["X_test"]).to(device)
    with torch.no_grad():
        y_pred = model(X_test).squeeze(-1).cpu().numpy()
    y_true = data["y_test"]
    test_rmse = rmse(y_true, y_pred)
    test_score = score(y_true, y_pred, SCORE_A, SCORE_B)

    metrics = {
        "model": args.model,
        "subset": args.subset,
        "n_features": data["n_features"],
        "window": data["window"],
        "clip_rul": data["clip_rul"],
        "n_params": n_params,
        "best_val_rmse": float(best_val_mse ** 0.5),
        "test_rmse": test_rmse,
        "test_score": test_score,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_history([h[0] for h in history], [h[1] for h in history], str(out_dir / "history.png"))
    plot_pred_true(y_true, y_pred, str(out_dir / "pred_vs_true.png"),
                   title=f"{args.model} {args.subset} · Pred vs True RUL")

    print(f"\n=== {args.model} {args.subset} 测试集 ===")
    print(f"  RMSE = {test_rmse:.3f}  cycles")
    print(f"  Score= {test_score:.1f}")
    print(f"  结果已保存至 {out_dir}")


if __name__ == "__main__":
    main()
