"""
预处理：读 train/ 分片 → 解析点击标签 → 截断行为序列 → 按频率挑 Top-K 物品建
「可训练物品 embedding 表」（用 SCL 多模态向量做预热初始化）→ 落盘。

【内存友好版】逐分片流式处理：
  - 只读需要的列（不读 151_2_180 类目序列等下游不用的重列）
  - 在 pyarrow 层面就把行为序列截断到 MAX_HIST（不把超长序列搬进 Python）
  - 每分片先落盘为「精简中间表」，最后再合并，避免一次性把 N 个分片
    完整数据 pd.concat 进内存（旧版会因此爆内存 / Segmentation fault）

输出（data/processed/）：
  train_prepared.parquet : 每行 (用户上下文 + 候选商品 + 历史序列 vocab 索引 + 点击)
  item_emb_init.npy     : (K+2, 128) float32，可训练物品表的「预热初始化」
                          行布局: 0=padding / 1=<unk>(稀有物品) / 2..K+1=Top-K 高频物品
  vocab_topk.npy        : (K,) int64，Top-K 物品的原始 id（升序，供查表）

【与冻结版的区别】旧版把 SCL 向量当「不可训练」的物品表征；本版只拿它做初始化，
真正训练时物品 embedding 由点击信号端到端更新——这是让 DIN 有望超过 LR 的关键。

用法：
  export N_SHARDS=1            # 先只跑 1 个分片验证 pipeline，再慢慢调大
  export EMB_VOCAB=500000     # 可训练物品表大小（高频物品数），按需调小省内存
  python prepare.py
"""
import os

import config
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from feature_emb import build_embeddings_for_ids

C = config.COL

# 只读取这些列（下游真正用到的），跳过 151_2_180 等下游不用的重型序列列
NEED = [
    C["label"], C["hist_item"],
    C["target_item"], C["target_cat"], C["target_city"], C["target_province"],
    C["user_id"], C["user_age"], C["user_gender"],
    C["user_province"], C["user_city"], C["user_city_level"],
]


def take_last_n(list_col, n):
    """取每条序列的「最后 n 条」（在 numpy 层面操作，内存安全，不把超长 list 转 Python）。

    用于验证 TAOBAO-MM 行为序列的存放方向：若 HIST_TAKE='last' 的 DIN AUC 明显更高，
    说明「最近行为在序列尾部」，应当保留末尾而非开头。
    """
    arr = list_col.combine_chunks() if hasattr(list_col, "combine_chunks") else list_col
    flat = pc.list_flatten(arr).to_numpy(zero_copy_only=False).astype(np.int64)
    offsets = arr.offsets.to_numpy().astype(np.int64)   # 长度 R+1
    starts, ends = offsets[:-1], offsets[1:]
    lengths = ends - starts
    row_start = np.maximum(0, lengths - n)              # 每条在自身切片内的起点
    abs_start = starts + row_start                     # flat 中的绝对起点
    R = len(starts)
    new_offsets = np.concatenate(([0], np.cumsum(ends - abs_start)))
    retained = np.concatenate([flat[int(s):int(e)] for s, e in zip(abs_start, ends)]) if R else np.empty(0, dtype=np.int64)
    return pa.ListArray.from_arrays(pa.array(new_offsets), pa.array(retained))


def process_shard(path, tmp_dir):
    """读 1 个分片 → 截断序列 → 解析标签 → 统计物品频率 → 写出精简中间表。

    全程在 pyarrow 层面操作，避免把超长 list 列转成 Python object 吞内存。
    返回 (中间表路径, 该分片 unique 物品 id, 对应出现次数)。
    """
    t = pq.read_table(path, columns=NEED)

    # 1) 行为序列截断：保留对点击判别最有用的那一端
    #    HIST_TAKE='first'（默认）：取前 MAX_HIST 条，假设「最近行为在最前」。
    #    HIST_TAKE='last'        ：取后 MAX_HIST 条（numpy 层实现，不爆内存），
    #                             用于验证序列方向。
    hi = t[C["hist_item"]]
    if config.HIST_TAKE == "last":
        hi_t = take_last_n(hi, config.MAX_HIST)
    else:
        hi_t = pc.list_slice(hi, 0, stop=config.MAX_HIST)
    t = t.set_column(t.schema.get_field_index(C["hist_item"]), C["hist_item"], hi_t)

    # 2) 解析点击标签：label_0 是长度2 one-hot，取第 2 个元素
    click = pc.list_element(t[C["label"]], 1).to_numpy(
        zero_copy_only=False).astype(np.int8)

    # 3) 统计本分片物品频率（候选 + 历史），用 numpy 而非 Python set
    target = t[C["target_item"]].to_numpy(zero_copy_only=False).astype(np.int64)
    hist_flat = pc.list_flatten(hi_t).to_numpy(
        zero_copy_only=False).astype(np.int64)
    all_dups = np.concatenate([target, hist_flat])
    shard_ids, shard_counts = np.unique(all_dups, return_counts=True)

    # 4) 组装精简表（标量列 + 截断后的序列 + 点击），落盘
    cols = {c: t[c] for c in NEED if c != C["label"]}
    cols["click"] = pa.array(click, type=pa.int8())
    out = pa.table(cols)
    out_path = os.path.join(tmp_dir, f"compact_{os.path.basename(path)}")
    pq.write_table(out, out_path)
    return out_path, shard_ids, shard_counts


def main():
    config.prepare_run_dir()
    config.save_run_config()
    files = config.shard_files()
    if not files:
        raise SystemExit("没有找到分片，请检查 TAOBAO_TRAIN_DIR / N_SHARDS")
    print(f"[cfg] 序列方向 HIST_TAKE={config.HIST_TAKE}  MAX_HIST={config.MAX_HIST}"
          f"  EMB_VOCAB={config.EMB_VOCAB}")
    tmp = os.path.join(config.PROCESSED_DIR, "_compact")
    os.makedirs(tmp, exist_ok=True)

    parts, id_parts, count_parts = [], [], []
    for i, f in enumerate(files):
        print(f"[load] 分片 {i + 1}/{len(files)}: {os.path.basename(f)}")
        p, ids, counts = process_shard(f, tmp)
        parts.append(p)
        id_parts.append(ids)
        count_parts.append(counts)

    # 合并所有分片的物品频率（向量化）
    all_ids = np.concatenate(id_parts)
    all_counts = np.concatenate(count_parts)
    order = np.argsort(all_ids, kind="stable")
    s_ids = all_ids[order]
    s_counts = all_counts[order]
    uniq_mask = np.empty(len(s_ids), dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = s_ids[1:] != s_ids[:-1]
    bounds = np.where(uniq_mask)[0]
    uniq_ids = s_ids[bounds]
    uniq_freq = np.add.reduceat(s_counts, bounds)
    print(f"[emb] 涉及 {len(uniq_ids):,} 个物品；总出现次数 {int(uniq_freq.sum()):,}")

    # 按频率取 Top-K 高频物品作为可训练词表（稀有物品统一映射到 <unk>）
    K = int(config.EMB_VOCAB)
    if len(uniq_ids) > K:
        topk_pos = np.argsort(uniq_freq)[::-1][:K]
        topk_ids = np.sort(uniq_ids[topk_pos])          # 升序，供 searchsorted 查表
    else:
        topk_ids = np.sort(uniq_ids)
        print(f"[emb] 物品数 {len(uniq_ids):,} <= EMB_VOCAB {K}，全部纳入可训练表")

    # 用 SCL 多模态向量给 Top-K 物品做「预热初始化」（mmap，不整体载入 4.3GB）
    emb_topk, _, found = build_embeddings_for_ids(
        topk_ids, config.EMB_KEYS, config.EMB_VALUES)
    hit_rate = float(found.mean()) if len(found) else 0.0
    print(f"[emb] SCL 向量命中率 {hit_rate:.3f}（未命中填 0，交由模型训练习得）")
    hit_mean = (emb_topk[found].mean(0)
                if found.any() else np.zeros(emb_topk.shape[1], np.float32))
    emb_init = np.zeros((len(topk_ids) + 2, emb_topk.shape[1]), dtype=np.float32)
    emb_init[1] = hit_mean          # 行 1 = <unk>（稀有物品），用命中向量均值预热
    emb_init[2:] = emb_topk         # 行 2.. = 高频物品 SCL 向量（warm start）

    item_emb_init_path = os.path.join(config.PROCESSED_DIR, "item_emb_init.npy")
    np.save(item_emb_init_path, emb_init)
    vocab_path = os.path.join(config.PROCESSED_DIR, "vocab_topk.npy")
    np.save(vocab_path, topk_ids)
    print(f"[emb] 可训练物品表 -> item_emb_init.npy  shape={emb_init.shape} "
          f"(0=padding, 1=<unk>, 2..K+1=Top-{len(topk_ids)})")

    # id -> vocab 索引：命中 Top-K → 行号+2；未命中 → 1(<unk>)；padding 用 0
    vocab_sorted = topk_ids

    def id_to_vidx(ids_arr):
        ids_arr = np.asarray(ids_arr, dtype=np.int64)
        pos = np.searchsorted(vocab_sorted, ids_arr)
        pos = np.clip(pos, 0, len(vocab_sorted) - 1)
        in_vocab = vocab_sorted[pos] == ids_arr
        return np.where(in_vocab, pos + 2, 1).astype(np.int64)

    # 合并精简中间表（此时已截断+去重列，体积远小于原始分片）
    print("[merge] 合并精简分片 ...")
    full = (pq.read_table(parts[0]) if len(parts) == 1
            else pa.concat_tables([pq.read_table(p) for p in parts]))
    df = full.to_pandas()

    # 采样（在映射之前，减少后续 Python 循环的行数）
    if config.SAMPLE_ROWS and config.SAMPLE_ROWS < len(df):
        df = df.sample(n=config.SAMPLE_ROWS, random_state=config.SEED).reset_index(drop=True)
        print(f"[sample] 采样到 {len(df):,} 行")

    # 物品 id → 可训练词表索引（候选 + 历史）
    df["target_vidx"] = id_to_vidx(df[C["target_item"]].values)
    df["hist_vidx"] = df[C["hist_item"]].apply(
        lambda h: id_to_vidx(np.asarray(h, dtype=np.int64)))

    df["hist_len"] = df[C["hist_item"]].apply(len)

    keep = [C["target_item"], "click", "hist_len", C["target_cat"], C["target_city"],
            C["target_province"], C["user_id"], C["user_age"],
            C["user_gender"], C["user_province"], C["user_city"],
            C["user_city_level"], "target_vidx", "hist_vidx"]
    out = df[keep]
    pp = os.path.join(config.PROCESSED_DIR, "train_prepared.parquet")
    out.to_parquet(pp, index=False)

    # 记录本次预处理的元信息，供 model_din.py 如实记录到指标里（防呆）
    import json as _json
    _meta = {"hist_take": config.HIST_TAKE, "max_hist": config.MAX_HIST,
             "emb_vocab": int(len(topk_ids)), "hit_rate": hit_rate,
             "emb_trainable": True}
    with open(os.path.join(config.PROCESSED_DIR, "prep_meta.json"), "w") as _f:
        _json.dump(_meta, _f, indent=2)

    # 清理中间表
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass
    print(f"[done] 训练表 -> {pp}  ({len(out):,} 行)")
    print(f"[done] 物品表初始化 -> {item_emb_init_path}  shape={emb_init.shape}")


if __name__ == "__main__":
    main()
