#!/bin/bash
# Taobao-MM CTR 预测 · 一键复现脚本
# 用法：bash run_all.sh
# 可通过环境变量覆盖默认配置，例如：RUN_ID=myexp N_SHARDS=2 bash run_all.sh

set -e  # 任何命令失败就退出

# ============ 配置（可通过环境变量覆盖）============
export RUN_ID=${RUN_ID:-reproduce}
export N_SHARDS=${N_SHARDS:-4}
export SAMPLE_ROWS=${SAMPLE_ROWS:-2000000}
export EMB_VOCAB=${EMB_VOCAB:-200000}
export MAX_HIST=${MAX_HIST:-500}
export HIST_TAKE=${HIST_TAKE:-first}
export EPOCHS=${EPOCHS:-6}
export BATCH=${BATCH:-512}
export DROPOUT=${DROPOUT:-0.1}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export SEED=${SEED:-42}

# ============ 检查环境变量 ============
if [ -z "$TAOBAO_TRAIN_DIR" ]; then
    echo "[错误] 请先设置 TAOBAO_TRAIN_DIR"
    echo "  export TAOBAO_TRAIN_DIR=/path/to/train"
    exit 1
fi

if [ -z "$TAOBAO_FEATUREMAP_DIR" ]; then
    echo "[错误] 请先设置 TAOBAO_FEATUREMAP_DIR"
    echo "  export TAOBAO_FEATUREMAP_DIR=/path/to/feature_map"
    exit 1
fi

# ============ 定位到脚本所在目录 ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/code"

# ============ 找到 Python ============
if [ -x "$SCRIPT_DIR/venv/Scripts/python.exe" ]; then
    PYTHON="$SCRIPT_DIR/venv/Scripts/python.exe"   # Windows
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"            # Linux/macOS
else
    PYTHON="python"
fi

echo "=========================================="
echo "Taobao-MM CTR 预测 · 一键复现"
echo "=========================================="
echo "RUN_ID       = $RUN_ID"
echo "N_SHARDS     = $N_SHARDS"
echo "SAMPLE_ROWS  = $SAMPLE_ROWS"
echo "EMB_VOCAB    = $EMB_VOCAB"
echo "MAX_HIST     = $MAX_HIST"
echo "HIST_TAKE    = $HIST_TAKE"
echo "EPOCHS       = $EPOCHS"
echo "BATCH        = $BATCH"
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
echo "Python       = $PYTHON"
echo "=========================================="

# ============ 1. 预处理 ============
echo ""
echo "[步骤 1/5] 预处理 ..."
"$PYTHON" prepare.py

# ============ 2. 基线 ============
echo ""
echo "[步骤 2/5] 训练基线（LR + LightGBM）..."
"$PYTHON" train_baseline.py

# ============ 3. DIN ============
echo ""
echo "[步骤 3/5] 训练 DIN ..."
"$PYTHON" model_din.py

# ============ 4. 离线推理验证 ============
echo ""
echo "[步骤 4/5] 离线推理验证 ..."
"$PYTHON" score.py

# ============ 5. 可视化 ============
echo ""
echo "[步骤 5/5] 生成可视化 ..."
"$PYTHON" plot_results.py

echo ""
echo "=========================================="
echo "全部完成"
echo "产物在 runs/$RUN_ID/"
echo "图片在 figures/"
echo "=========================================="
