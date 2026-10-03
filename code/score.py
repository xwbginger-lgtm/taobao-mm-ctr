# -*- coding: utf-8 -*-
"""
离线推理验证：加载训练好的 best_din.pt，对验证集样本独立打分，
确认 AUC 与训练记录一致。
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import config


RUN_ID = os.environ["RUN_ID"]  # 必须显式指定，比如 r008_4shards
RUN_DIR = os.path.join(config.RUNS_DIR, RUN_ID)
PROCESSED = os.path.join(RUN_DIR, "processed")
DIN_OUT = os.path.join(RUN_DIR, "outputs", "din")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DIN(nn.Module):
    """和 model_din.py 里结构完全一致，只用于加载权重。"""

    def __init__(self, num_items, emb_dim, cat_cardinalities, hidden=128, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.item_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(c + 1, 8) for c in cat_cardinalities]
        )
        ctx_dim = 8 * len(cat_cardinalities) + 1
        self.attn = nn.Sequential(
            nn.Linear(emb_dim * 4, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + ctx_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, tgt_vidx, hist_vidx, length_mask, hist_len, cat_idx):
        tgt = self.item_proj(self.item_emb(tgt_vidx))
        hist = self.item_proj(self.item_emb(hist_vidx))
        q = tgt.unsqueeze(1).expand_as(hist)
        a = self.attn(torch.cat([q, hist, q - hist, q * hist], -1)).squeeze(-1)
        a = a.masked_fill(length_mask == 0, -1e9)
        w = torch.softmax(a, -1) * length_mask
        w = (w / w.sum(-1, keepdim=True).clamp_min(1e-9)).unsqueeze(-1)
        interest = (w * hist).sum(1)
        cat_e = torch.cat(
            [self.cat_emb[i](cat_idx[:, i]) for i in range(len(self.cat_emb))], -1
        )
        ctx = torch.cat([cat_e, hist_len], -1)
        return self.mlp(torch.cat([interest, tgt, ctx], -1)).squeeze(-1)


def main():
    # 1) 读模型配置
    with open(os.path.join(DIN_OUT, "model_config.json"), encoding="utf-8") as f:
        mc = json.load(f)

    num_items = mc["num_items"]
    emb_dim = mc["emb_dim"]
    cat_cardinalities = mc["cat_cardinalities"]
    cat_cols = mc["cat_cols"]
    max_hist = mc["max_hist"]

    print(f"[config] RUN_ID={RUN_ID}")
    print(f"[config] num_items={num_items}, emb_dim={emb_dim}, max_hist={max_hist}")
    print(f"[config] cat_cardinalities={cat_cardinalities}")

    # 2) 加载编码器
    with open(os.path.join(DIN_OUT, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    print(f"[config] encoders loaded: {list(encoders.keys())}")

    # 3) 加载数据，用训练时相同的划分取验证集
    pp = os.path.join(PROCESSED, "train_prepared.parquet")
    df = pd.read_parquet(pp)
    split = np.load(os.path.join(DIN_OUT, "split.npz"))
    va_idx = split["valid"]
    va = df.iloc[va_idx].reset_index(drop=True)
    print(f"[data] 验证集 {len(va):,} 行")

    # 4) 构造输入张量
    y_true = va["click"].astype(int).values

    # hist_vidx 填充到 max_hist
    def pad_hist(h):
        h = list(h)[:max_hist]
        L = len(h)
        return np.array(h + [0] * (max_hist - L), dtype=np.int64), L

    hist_list, len_list = [], []
    for h in va["hist_vidx"]:
        padded, L = pad_hist(h)
        hist_list.append(padded)
        len_list.append(L)

    hist_vidx = torch.tensor(np.stack(hist_list), dtype=torch.long)
    length_mask = torch.tensor(
        np.array([[1] * L + [0] * (max_hist - L) for L in len_list], dtype=np.float32)
    )
    hist_len = torch.tensor(
        np.array(len_list, dtype=np.float32).reshape(-1, 1) / max_hist,
        dtype=torch.float32,
    )

    tgt_vidx = torch.tensor(va["target_vidx"].values, dtype=torch.long)

    cat_idx = np.stack([
        va[c].astype(str).map(encoders[c]).fillna(0).astype(np.int64).values
        for c in cat_cols
    ], axis=1)
    cat_idx = torch.tensor(cat_idx, dtype=torch.long)

    # 5) 加载模型
    model = DIN(num_items, emb_dim, cat_cardinalities).to(DEVICE)
    state = torch.load(
        os.path.join(DIN_OUT, "best_din.pt"),
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    print(f"[model] best_din.pt loaded, 参数在 {DEVICE}")

    # 6) 批量打分
    batch_size = 2048
    preds = []
    with torch.no_grad():
        for i in range(0, len(va), batch_size):
            sl = slice(i, i + batch_size)
            logits = model(
                tgt_vidx[sl].to(DEVICE),
                hist_vidx[sl].to(DEVICE),
                length_mask[sl].to(DEVICE),
                hist_len[sl].to(DEVICE),
                cat_idx[sl].to(DEVICE),
            )
            preds.append(logits.cpu().numpy())
    preds = np.concatenate(preds)

    # 7) 计算指标
    auc = roc_auc_score(y_true, preds)
    print(f"\n[result] 验证集 AUC = {auc:.4f}")
    print(f"[result] 正样本平均分 = {preds[y_true == 1].mean():.4f}")
    print(f"[result] 负样本平均分 = {preds[y_true == 0].mean():.4f}")

    # 8) 和训练记录对比
    metrics_path = os.path.join(DIN_OUT, "din_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        train_auc = m.get("valid_auc")
        if train_auc:
            diff = abs(auc - train_auc)
            print(f"[check] 训练记录 AUC = {train_auc:.4f}, 差异 = {diff:.6f}")
            if diff < 0.001:
                print("[check] PASS: 推理 AUC 与训练记录一致")
            else:
                print("[check] WARN: 差异偏大，检查推理路径是否和训练一致")

    print("\n[done] 离线推理验证完成")


if __name__ == "__main__":
    main()