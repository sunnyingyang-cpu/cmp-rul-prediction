# 环境搭建与数据准备

## 1. Python 环境
- 推荐 Python 3.10（本项目在 3.10 验证）。
- 建议使用虚拟环境 / conda 隔离：
  ```bash
  conda create -n cmp python=3.10
  conda activate cmp
  pip install -r requirements.txt
  ```
- 核心依赖：torch（CPU 版即可跑通；有 CUDA 则训练更快）、numpy、pandas、
  scikit-learn、matplotlib、tqdm、onnx、onnxruntime、flask。

## 2. 获取数据集（C-MAPSS）
- 公开镜像（已校验，含全部 12 个 txt）：
  [github.com/NicolasDeHorta/CMAPSSData](https://github.com/NicolasDeHorta/CMAPSSData)
- 把以下文件放到 `datasets/` 目录（文件名必须一致）：
  ```
  train_FD001.txt  test_FD001.txt  RUL_FD001.txt
  train_FD002.txt  test_FD002.txt  RUL_FD002.txt
  train_FD003.txt  test_FD003.txt  RUL_FD003.txt
  train_FD004.txt  test_FD004.txt  RUL_FD004.txt
  ```
- `datasets/` 已在 `.gitignore` 中排除，不会进入版本库（保持仓库轻量）。

## 3. 环境自检（推荐先跑）
```bash
python tools/check_env.py
```
秒级完成，不训练。它会校验：
1. 依赖包（torch/numpy/pandas/sklearn/matplotlib/tqdm 必需；onnx/onnxruntime/flask 可选）
2. `datasets/` 下 12 个 txt 是否齐全、列数是否为 26
3. **每个子集「测试引擎数 == RUL 行数」**（换数据源后最易静默错位的点）
4. 各子集特征选择结果
5. 四类模型能否正常构建并完成一次前向推理

全部 `[OK]` 即环境就绪。

## 4. 跑通最小样例
```bash
python tools/train.py --model lstm --subset FD001 --device cpu --epochs 1
```
若无报错并生成 `runs/lstm_FD001/`，说明环境就绪。

## 5. GPU 加速（可选）
- 本机有 CUDA 时，把 `--device cuda` 即可：
  ```bash
  python tools/train.py --model transformer --subset FD001 --device cuda
  ```
- 注意：CUDA 版 PyTorch 需单独安装（非 `requirements.txt` 默认的 CPU 版）。

## 6. 目录说明
- `runs/`：训练产物（checkpoint / 指标 / 曲线图），`.gitignore` 排除。
- `weights/`：导出的 ONNX 模型，`.gitignore` 排除。
- `docs/figures/`：说明文档配图，**入库**（README 引用）。
