"""
C-MAPSS 设备剩余寿命（RUL）预测 — 全局配置与常量。

所有超参集中在此，方便复现与调优。训练脚本（tools/train.py）会以命令行参数覆盖这里的部分默认值。
"""

# C-MAPSS 四个子集（FD = Fleet Data）
FD_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# 数据列：unit(1) cycle(1) + 3 个工况设置 + 21 个传感器 = 26 列
INDEX_NAMES = ["unit", "cycle"]
SETTING_NAMES = ["setting1", "setting2", "setting3"]
SENSOR_NAMES = [f"sensor{i}" for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

# 数据预处理默认参数
WINDOW = 30          # 滑动窗口长度（用最近 N 个循环预测 RUL）
CLIP_RUL = 125       # RUL 标签上限裁剪（稳定训练，避免早期健康期巨大 RUL 主导损失）
STRIDE = 1           # 训练滑动窗口步长
VAL_RATIO = 0.2      # 从训练集按 engine 切分验证集比例
VAR_THRESHOLD = 1e-6 # 传感器方差阈值，低于此值视为恒定特征并剔除

# 评估指标（NASA 竞赛定义）
SCORE_A = 13.0       # 早预测（pred < true）的衰减系数
SCORE_B = 10.0       # 晚预测（pred > true）的衰减系数

# 模型默认超参（可通过命令行覆盖）
MODEL_DEFAULTS = {
    "lstm":      dict(hidden=64, num_layers=2, dropout=0.2, bidirectional=False),
    "bilstm":    dict(hidden=64, num_layers=2, dropout=0.2, bidirectional=True),
    "cnn":       dict(channels=32, kernel=3, dropout=0.2),
    "transformer": dict(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1),
}

# 训练默认超参
TRAIN_DEFAULTS = dict(
    epochs=80,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    patience=12,        # 验证 RMSE 早停耐心
    clip_grad=1.0,
    seed=42,
)
