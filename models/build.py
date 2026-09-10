"""
RUL 预测模型定义：四条技术路线。

输入：张量 (B, T, F) —— B 个样本，每条为最近 T=window 个循环、F 个特征。
输出：张量 (B, 1) —— 每个样本预测的剩余寿命（RUL，单位：循环）。

- RUL_LSTM / RUL_BiLSTM：循环神经网络，取最后一个时间步隐状态回归。
- RUL_CNN：1D 时序卷积（沿时间维卷积），全局池化后回归。
- RUL_Transformer：轻量 Transformer 编码器，取最后时间步回归。

build_model(name, n_features, **kwargs) 统一工厂入口。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RUL_LSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, hidden*dir)
        last = out[:, -1, :]           # 取最后时间步
        return self.head(last)


class RUL_CNN(nn.Module):
    def __init__(self, n_features: int, channels: int = 32, kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, padding=kernel // 2),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels * 2, kernel, padding=kernel // 2),
            nn.BatchNorm1d(channels * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # (B, channels*2, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, F, T)
        feat = self.block(x).squeeze(-1)
        return self.head(feat)


class RUL_Transformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)         # (B, T, d_model)
        out = self.encoder(x)
        last = out[:, -1, :]
        return self.head(last)


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("lstm", "bilstm"):
        bidirectional = (name == "bilstm")
        kw = {k: v for k, v in kwargs.items() if k in ("hidden", "num_layers", "dropout")}
        return RUL_LSTM(n_features, bidirectional=bidirectional, **kw)
    if name == "cnn":
        kw = {k: v for k, v in kwargs.items() if k in ("channels", "kernel", "dropout")}
        return RUL_CNN(n_features, **kw)
    if name == "transformer":
        kw = {k: v for k, v in kwargs.items()
              if k in ("d_model", "nhead", "num_layers", "dim_feedforward", "dropout")}
        return RUL_Transformer(n_features, **kw)
    raise ValueError(f"未知模型: {name}")
