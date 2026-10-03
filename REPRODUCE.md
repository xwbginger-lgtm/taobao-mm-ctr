# 复现指南

本文件说明如何在本地复现本项目的实验结果。

## 一、环境要求

### 硬件

| 项目 | 最低要求 | 本项目实际使用 |
|---|---|---|
| 显存 | 4GB | 6GB |
| 内存 | 16GB | 16GB |
| 磁盘 | 10GB 可用 | 51GB 可用 |
| GPU | NVIDIA（CUDA） | 是 |

**如果显存不足**，调小 `BATCH`、`MAX_HIST`、`SAMPLE_ROWS` 或 `N_SHARDS`。

### 软件

| 项目 | 版本 |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.0+ |
| CUDA | 11.8 或 12.1 |
| 操作系统 | Windows 10/11、Linux、macOS |

### 安装依赖

```bash
pip install -r requirements.txt
```

如果 requirements.txt 是从 pip freeze 导出的，可能包含多余包。核心依赖是：

```
torch>=2.0
numpy
pandas
pyarrow
scikit-learn
lightgbm
matplotlib
```

## 二、数据准备

从 Hugging Face 下载 Taobao-MM 数据集，只需要 `train/` 和 `feature_map/` 两部分。

### 目录结构

```
你的数据目录/
├── train/
│   ├── train-shard-000000.parquet
│   ├── train-shard-000001.parquet
│   └── ...（共 161 个分片）
└── feature_map/
    ├── scl_emb_int8_p90_keys.npy
    └── scl_emb_int8_p90_values.npy
```

### 设置环境变量

Git Bash / Linux / macOS：

```bash
export TAOBAO_TRAIN_DIR=/path/to/train
export TAOBAO_FEATUREMAP_DIR=/path/to/feature_map
```

PowerShell：

```powershell
$env:TAOBAO_TRAIN_DIR = "E:\taobao-mm\train"
$env:TAOBAO_FEATUREMAP_DIR = "E:\taobao-mm\feature_map"
```

## 三、一键复现

### 方式 A：用 run_all.sh（推荐）

```bash
cd /e/taobao-mm
bash run_all.sh
```

### 方式 B：手动逐条跑

```bash
cd code

export RUN_ID=reproduce
export N_SHARDS=4
export SAMPLE_ROWS=2000000
export EMB_VOCAB=200000
export MAX_HIST=500
export HIST_TAKE=first
export EPOCHS=6
export BATCH=512
export DROPOUT=0.1
export WEIGHT_DECAY=1e-4

python prepare.py
python train_baseline.py
python model_din.py
python score.py
python plot_results.py
```

> **注意**：RUN_ID 必须是新目录。如果 `runs/reproduce/` 已存在，`prepare.py` 会报错。换一个 RUN_ID 或删掉旧目录。

## 四、预期输出

跑完后，你应该看到以下关键输出。如果你的输出和这里差异很大，说明复现出了问题。

### prepare.py

```
[cfg] 序列方向 HIST_TAKE=first  MAX_HIST=500  EMB_VOCAB=200000
[load] 分片 1/4: train-shard-000000.parquet
[load] 分片 2/4: train-shard-000001.parquet
[load] 分片 3/4: train-shard-000002.parquet
[load] 分片 4/4: train-shard-000003.parquet
[emb] 涉及 22,486,352 个物品；总出现次数 991,812,430
[emb] SCL 向量命中率 1.000
[emb] 可训练物品表 -> item_emb_init.npy  shape=(200002, 128)
[done] 训练表 -> runs/reproduce/processed/train_prepared.parquet  (2,000,000 行)
```

关键数字：

- 物品种类约 2200 万
- 总出现次数约 9.9 亿
- SCL 命中率 1.000
- 训练表 200 万行

### train_baseline.py

```
[data]  train=1,800,000  valid=200,000  CTR=0.1375
[base]  盲猜            AUC=0.5000  LogLoss=0.4004
[lr]    LogisticRegression  AUC=0.6235  LogLoss=0.7007
[lgb]   LightGBM          AUC=0.5792  LogLoss=1.0436
```

关键数字：

- CTR 约 13.75%
- LR AUC 约 0.62
- LightGBM AUC 约 0.58

### model_din.py

```
[info] 设备=cuda  数据=2,000,000 行  序列长度=500  批大小=512
[model] 可训练物品表 200,002 项 × 128 维
[epoch 0] train_loss=1.1806  valid AUC=0.5876
[epoch 1] train_loss=1.1664  valid AUC=0.5931
[epoch 2] train_loss=1.1526  valid AUC=0.6292
[epoch 3] train_loss=1.1173  valid AUC=0.6399
[epoch 4] train_loss=1.0955  valid AUC=0.6426
[epoch 5] train_loss=1.0832  valid AUC=0.6416
[done] DIN 在 cuda 上训完（完成），valid AUC=0.6426
```

关键数字：

- 最终 valid AUC 约 0.64
- epoch 4 后基本收敛

### score.py

```
[config] RUN_ID=reproduce
[data] 验证集 200,000 行
[result] 验证集 AUC = 0.6426
[result] 正样本平均分 = 0.1983
[result] 负样本平均分 = -0.1178
[check] 训练记录 AUC = 0.6426, 差异 = 0.000000
[check] PASS: 推理 AUC 与训练记录一致
```

关键：差异 < 0.001，说明推理路径正确。

### plot_results.py

生成 8 张图到 `figures/` 目录。

## 五、常见问题

**Q1：TRAIN_DIR 不存在**

原因：没有设 `TAOBAO_TRAIN_DIR` 环境变量。

解决：

```bash
export TAOBAO_TRAIN_DIR=/path/to/train
```

**Q2：请显式设置 RUN_ID**

原因：`RUN_ID` 是必设的环境变量。

解决：

```bash
export RUN_ID=my_experiment
```

**Q3：FileExistsError: runs/xxx**

原因：RUN_ID 对应的目录已存在。

解决：换一个 RUN_ID，或删掉旧目录：

```bash
rm -rf runs/旧RUN_ID
```

**Q4：CUDA out of memory**

原因：显存不足。

解决：调小以下参数：

```bash
export BATCH=256        # 或 128
export MAX_HIST=200     # 或 100
export SAMPLE_ROWS=1000000
export N_SHARDS=2
```

**Q5：Unable to allocate X KiB**

原因：系统内存不足，通常发生在 `prepare.py` 合并多个分片时。

解决：减少 `N_SHARDS`（比如从 4 降到 2）。

**Q6：AUC 和预期差很多**

可能原因：

- 数据分片不同（`N_SHARDS` 不同）
- 采样行数不同（`SAMPLE_ROWS` 不同）
- 随机种子不同（默认 `SEED=42`）
- 训练轮数不同（`EPOCHS` 不同）

解决：确认以上参数和本指南一致。

**Q7：跑得很慢**

原因：可能没用到 GPU。

检查：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出 `False`，说明 PyTorch 是 CPU 版，或 CUDA 配置有问题。

## 六、复现的边界

**能复现的**：

- 代码逻辑和实验流程
- 在当前数据子集上的相对结论（DIN > LR > LightGBM）
- 超参消融的趋势（weight_decay、词表大小的影响）

**不能完全复现的**：

- 绝对 AUC 数字（取决于数据分片、硬件、随机性）
- 全量 161 个分片的结果（本项目受硬件限制只跑了 4 个）

## 七、项目边界

本项目聚焦精排阶段的 CTR 预测。Taobao-MM 数据集是离线快照，不含实时行为流、召回层和物品实时特征，因此不适用于端到端实时推荐系统。

本项目验证的是“DIN + SCL 多模态 embedding 在精排任务上的效果”，不涉及召回、冷启动、线上服务。
