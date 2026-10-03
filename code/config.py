"""
TAOBAO-MM 简历项目 · 配置（针对 train/ + feature_map/ 格式）

你本地只有两个目录：
  - train/        52.6 GB，分片 parquet（已 join 好的训练表，每行=用户+候选商品+标签）
  - feature_map/   7.11 GB，.npy（SCL 多模态向量 + 各特征词表映射）

用法：把下面两个路径通过环境变量传进来（或改这里的默认值）：
  export TAOBAO_TRAIN_DIR=/你的路径/train
  export TAOBAO_FEATUREMAP_DIR=/你的路径/feature_map
"""
import json
import os
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ID = os.environ.get("RUN_ID", "run_dev")
if not RUN_ID or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in RUN_ID):
    raise ValueError("RUN_ID 只能包含字母、数字、连字符和下划线")

RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
PROCESSED_DIR = os.path.join(RUN_DIR, "processed")
OUT_DIR = os.path.join(RUN_DIR, "outputs")
RUN_META_PATH = os.path.join(RUN_DIR, "run_config.json")


def prepare_run_dir(stage="prepare"):
    global OUT_DIR
    if "RUN_ID" not in os.environ:
        raise ValueError("请显式设置 RUN_ID，避免误用默认实验目录")
    if stage == "prepare":
        os.makedirs(RUNS_DIR, exist_ok=True)
        os.mkdir(RUN_DIR)
        os.mkdir(PROCESSED_DIR)
        os.mkdir(os.path.join(RUN_DIR, "outputs"))
    else:
        with open(RUN_META_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        for key in ("max_hist", "hist_take", "seed"):
            if saved[key] != globals()[key.upper()]:
                raise ValueError(f"当前 {key} 与预处理配置不同，请恢复本轮配置")
        if not os.path.isfile(os.path.join(PROCESSED_DIR, "prep_meta.json")):
            raise RuntimeError("预处理尚未完成")
        OUT_DIR = os.path.join(RUN_DIR, "outputs", stage)
        os.mkdir(OUT_DIR)
    import shutil
    snapshot = os.path.join(RUN_DIR, "source" if stage == "prepare" else f"source_{stage}")
    os.mkdir(snapshot)
    for name in ("config.py", "prepare.py", "model_din.py", "train_baseline.py", "feature_emb.py", "requirements.txt"):
        shutil.copy2(os.path.join(PROJECT_DIR, name), snapshot)


def save_run_config():
    if os.path.exists(RUN_META_PATH):
        return

    payload = {
        "run_id": RUN_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_dir": TRAIN_DIR,
        "featuremap_dir": FEATUREMAP_DIR,
        "n_shards": N_SHARDS,
        "max_hist": MAX_HIST,
        "emb_vocab": EMB_VOCAB,
        "hist_take": HIST_TAKE,
        "sample_rows": SAMPLE_ROWS,
        "seed": SEED,
    }
    with open(RUN_META_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

# ↓↓↓ 改成你本地的两个目录，或设置环境变量 ↓↓↓
TRAIN_DIR = os.environ.get("TAOBAO_TRAIN_DIR", "/path/to/Taobao-MM/train")
FEATUREMAP_DIR = os.environ.get(
    "TAOBAO_FEATUREMAP_DIR", "/path/to/Taobao-MM/feature_map"
)

# 快速跑：只读前 N 个分片（每个 ~330MB）。先验证 pipeline 用 2~3 个，
# 之后想认真训可以把 N_SHARDS 调大或设成 None（全量 161 个分片）。
N_SHARDS = int(os.environ.get("N_SHARDS", "3"))

# 行为序列截断长度（原始最长 1000；取 500 给 DIN 更多兴趣上下文）
MAX_HIST = int(os.environ.get("MAX_HIST", "500"))

# 可训练物品 embedding 的词表上限（按出现频率取 Top-K，稀有物品归并 <unk>）。
# 不训全量 800 万物品（会爆内存），50 万高频物品参数量约 256MB，笔记本可跑。
EMB_VOCAB = int(os.environ.get("EMB_VOCAB", "500000"))

# 行为序列保留哪一端（用于验证 TAOBAO-MM 序列的存放方向）：
#   "first" (默认) -> 保留前 MAX_HIST 条，假设「最近行为在最前」
#   "last"         -> 保留后 MAX_HIST 条，假设「最近行为在最后」
# 切换 HIST_TAKE 各跑一遍 DIN，对比 valid AUC 即可判断方向。
HIST_TAKE = os.environ.get("HIST_TAKE", "first").lower()

# 基线训练采样行数（None = 用所选分片全量）
SAMPLE_ROWS = int(os.environ.get("SAMPLE_ROWS", "500000"))

SEED = 42

# ---- 列名（来自官方 README，已确认）----
COL = {
    "user_id": "129_1",
    "label": "label_0",            # 长度2 one-hot: [1,0]未点 / [0,1]点击
    "hist_item": "150_2_180",      # 历史行为 item id 序列
    "hist_cat": "151_2_180",       # 历史行为 item 类目序列
    "target_item": "205",          # 候选商品 id
    "target_cat": "206",           # 候选商品 类目
    "target_city": "213",          # 候选商品 市
    "target_province": "214",      # 候选商品 省
    "user_age": "130_1",
    "user_gender": "130_2",
    "user_province": "130_3",
    "user_city": "130_4",
    "user_city_level": "130_5",
}
# 需要「标签编码」的类别列
CAT_COLS = [
    COL["user_age"], COL["user_gender"], COL["user_province"],
    COL["user_city"], COL["user_city_level"],
    COL["target_cat"], COL["target_city"], COL["target_province"],
]

# ---- feature_map 里的多模态向量 ----
EMB_KEYS = os.path.join(FEATUREMAP_DIR, "scl_emb_int8_p90_keys.npy")     # 物品 id
EMB_VALUES = os.path.join(FEATUREMAP_DIR, "scl_emb_int8_p90_values.npy") # (N,128) int8


def shard_files():
    """返回 train/ 下排序后的分片路径（只取前 N_SHARDS 个）。"""
    if not os.path.isdir(TRAIN_DIR):
        raise SystemExit(f"TRAIN_DIR 不存在: {TRAIN_DIR}\n请设置环境变量 TAOBAO_TRAIN_DIR")
    fs = sorted(
        f for f in os.listdir(TRAIN_DIR)
        if f.startswith("train-shard") and f.endswith(".parquet")
    )
    if N_SHARDS is not None:
        fs = fs[:N_SHARDS]
    return [os.path.join(TRAIN_DIR, f) for f in fs]
