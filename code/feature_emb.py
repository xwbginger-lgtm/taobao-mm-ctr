"""
读取 feature_map/ 里的 SCL 多模态向量（int8 量化，128 维）。

文件：
  scl_emb_int8_p90_keys.npy   : (N,) int64   物品 id
  scl_emb_int8_p90_values.npy : (N,128) int8 对应向量（mmap，不整体载入内存）

对外只暴露 build_embeddings_for_ids()：给一批物品 id，返回 (n,128) float32
矩阵 + id→行号 映射。用「排序 + searchsorted」定位，缺失物品填 0 向量，
避免把 4.3GB 整个读进内存。
"""
import numpy as np


def build_embeddings_for_ids(item_ids, keys_path, values_path, scale=127.0):
    """
    参数
      item_ids : array-like of int   想要向量的物品 id
    返回
      emb      : (len(item_ids), 128) float32，缺失的填 0
      id2row   : {item_id: 行号}
      found    : (len(item_ids),) bool，是否命中（覆盖 100% 候选/90% 行为物品）
    """
    keys = np.load(keys_path)                       # (N,) int64
    values = np.load(values_path, mmap_mode="r")    # (N,128) int8, 内存映射
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    q = np.asarray(item_ids, dtype=keys.dtype)
    pos = np.searchsorted(sorted_keys, q)
    pos = np.clip(pos, 0, len(sorted_keys) - 1)
    found = sorted_keys[pos] == q

    emb = np.zeros((len(q), values.shape[1]), dtype=np.float32)
    hit = order[pos]                                # 命中的原始行号
    emb[found] = values[hit[found]].astype(np.float32) / scale

    id2row = {int(i): r for r, i in enumerate(q)}
    return emb, id2row, found
