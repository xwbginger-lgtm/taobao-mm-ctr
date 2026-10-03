"""
【进阶】DIN 风格「目标注意力」序列模型 —— 用一张**可训练的物品 embedding 表**
（以 SCL 多模态向量预热初始化，再由点击信号端到端更新）作为 item 表示，
对用户的超长行为历史做 target attention，预测点击。

亮点（简历可直接写）：
  1) 物品表征不是冻结的——用 SCL 128 维多模态向量做 warm start，再用点击标签
     端到端训练，让模型自己学到「哪些物品与当前候选更相关」；
  2) 高频物品进可训练表（Top-K），稀有物品归并 <unk>，padding 单独占位，
     在有限显存/内存下也能训练大规模物品表征；
  3) 类别特征走 Embedding，不再把裸整数编码直接塞进 MLP；
  4) 用经典 DIN 局部激活单元 [query, key, query-key, query*key] 计算注意力。

数据：data/processed/train_prepared.parquet + item_emb_init.npy（prepare.py 产出）
用法：  python prepare.py && python model_din.py
进阶：  EPOCHS=30 BATCH=512 python model_din.py
第三轮（更多数据 + 更强正则）：
  N_SHARDS=2 SAMPLE_ROWS=1000000 python prepare.py
  WEIGHT_DECAY=1e-4 DROPOUT=0.1 python model_din.py
"""
import json
import os
import time

import config
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

C = config.COL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH = int(os.environ.get("BATCH", "512"))
# 第三轮：加强正则以对抗 epoch-1 过拟合（均可通过环境变量覆盖）
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))


class TaobaoDS(Dataset):
    """每行 = (候选 vocab 索引, 历史 vocab 索引序列, 长度掩码, 类别索引, 点击标签)。"""

    def __init__(self, df, encoders):
        self.df = df.reset_index(drop=True)
        self.enc = encoders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        hist = list(r["hist_vidx"])[: config.MAX_HIST]
        L = len(hist)
        pad = config.MAX_HIST - L
        hidx = np.array(hist + [0] * pad, dtype=np.int64)   # 0 = padding
        length_mask = np.array([1] * L + [0] * pad, dtype=np.float32)
        cat_idx = np.array(
            [self.enc[c].get(str(r[c]), 0) for c in config.CAT_COLS], dtype=np.int64
        )
        return int(r["target_vidx"]), hidx, length_mask, cat_idx, int(r["click"])


def collate(batch):
    """直接返回整数索引，embedding 查表放在模型里（可训练，需梯度）。"""
    tgts = np.array([b[0] for b in batch])
    hidx = np.stack([b[1] for b in batch])
    length_mask = np.stack([b[2] for b in batch])
    hist_len = length_mask.sum(1, keepdims=True) / config.MAX_HIST   # (B,1) 归一化长度
    cat_idx = np.stack([b[3] for b in batch])
    ys = np.array([b[4] for b in batch])
    return (torch.tensor(tgts), torch.tensor(hidx),
            torch.tensor(length_mask, dtype=torch.float32),
            torch.tensor(hist_len, dtype=torch.float32),
            torch.tensor(cat_idx),
            torch.tensor(ys, dtype=torch.float32))


class DIN(nn.Module):
    def __init__(self, num_items, emb_dim, cat_cardinalities, hidden=128, dropout=0.1):
        super().__init__()
        # 可训练物品 embedding：SCL 向量预热初始化，点击信号端到端更新
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.item_emb.weight.requires_grad_(True)
        # 额外可训练投影，给表征再叠一层任务适配（可选但增容）
        self.item_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.item_proj.weight)
        # 类别特征走 Embedding（每个类别维度 8），避免裸整数编码污染 MLP
        self.cat_emb = nn.ModuleList(
            [nn.Embedding(c + 1, 8) for c in cat_cardinalities]
        )
        ctx_dim = 8 * len(cat_cardinalities) + 1        # +1 为归一化后的 hist_len
        # 注意力：DIN 局部激活单元 [query, key, query-key, query*key]
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
        tgt = self.item_proj(self.item_emb(tgt_vidx))        # (B, D)
        hist = self.item_proj(self.item_emb(hist_vidx))       # (B, T, D)
        q = tgt.unsqueeze(1).expand_as(hist)                  # (B, T, D)
        # 局部激活单元
        a = self.attn(torch.cat([q, hist, q - hist, q * hist], -1)).squeeze(-1)  # (B, T)
        # 只屏蔽 padding（length_mask），<unk>(idx=1) 是真实稀有物品，保留参与注意力
        a = a.masked_fill(length_mask == 0, -1e9)
        w = torch.softmax(a, -1) * length_mask
        w = (w / w.sum(-1, keepdim=True).clamp_min(1e-9)).unsqueeze(-1)
        interest = (w * hist).sum(1)                          # (B, D)
        # 类别 embedding + 历史长度
        cat_e = torch.cat(
            [self.cat_emb[i](cat_idx[:, i]) for i in range(len(self.cat_emb))], -1
        )
        ctx = torch.cat([cat_e, hist_len], -1)                # (B, 8*n_cat + 1)
        # 输出原始 logits（配合 BCEWithLogitsLoss，支持类不平衡加权）
        return self.mlp(torch.cat([interest, tgt, ctx], -1)).squeeze(-1)


def main():
    config.prepare_run_dir("din")
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    pp = os.path.join(config.PROCESSED_DIR, "train_prepared.parquet")
    ep = os.path.join(config.PROCESSED_DIR, "item_emb_init.npy")
    if not (os.path.exists(pp) and os.path.exists(ep)):
        raise SystemExit("先跑 python prepare.py（新版本会产出 item_emb_init.npy）")
    df = pd.read_parquet(pp)
    if "hist_vidx" not in df.columns:
        raise SystemExit(
            "train_prepared.parquet 仍是旧版（缺 hist_vidx 列）。\n"
            "新版 prepare.py 改变了输出格式，请先重跑：python prepare.py")
    emb_init = np.load(ep)

    # —— 速度/资源自检（开头就打印，方便判断为什么慢）——
    n_cpu = os.cpu_count() or 4
    n_thr = torch.get_num_threads()
    print(f"[info] 设备={DEVICE}  线程数={n_thr} (物理核 {n_cpu})  "
          f"数据={len(df):,} 行  序列长度={config.MAX_HIST}  "
          f"批大小={BATCH}  epoch={NUM_EPOCHS}  "
          f"weight_decay={WEIGHT_DECAY}  dropout={DROPOUT}")
    if DEVICE == "cpu":
        if n_thr < n_cpu:
            print(f"[warn] 当前只用 {n_thr} 线程 < 物理核 {n_cpu}！"
                  f"运行前先执行：  export OMP_NUM_THREADS={n_cpu}")
        print("[warn] 未检测到 GPU，全程 CPU 训练。DIN 是计算密集型，会极慢；"
              "建议：①设满线程数 ②减 MAX_HIST/ SAMPLE_ROWS ③换 GPU 版 PyTorch。")
    num_items, emb_dim = emb_init.shape
    config.EMB_DIM = emb_dim
    print(f"[model] 可训练物品表 {num_items:,} 项 × {emb_dim} 维（SCL 预热，端到端训练）")

    y = df["click"].astype(int).values
    tr_idx, va_idx = train_test_split(
        np.arange(len(df)), test_size=0.1, random_state=config.SEED, stratify=y
    )
    tr, va = df.iloc[tr_idx], df.iloc[va_idx]

    # 类别编码：0 留给 unknown，训练集上 fit
    encoders, cardinalities = {}, []
    for c in config.CAT_COLS:
        le = LabelEncoder()
        le.fit(tr[c].astype(str))
        mapping = {v: i + 1 for i, v in enumerate(le.classes_)}
        encoders[c] = mapping
        cardinalities.append(len(le.classes_) + 1)

    import pickle
    with open(os.path.join(config.OUT_DIR, "encoders.pkl"), "xb") as f:
        pickle.dump(encoders, f)
    np.savez(os.path.join(config.OUT_DIR, "split.npz"), train=tr_idx, valid=va_idx)
    with open(os.path.join(config.OUT_DIR, "model_config.json"), "x", encoding="utf-8") as f:
        json.dump({"run_id": config.RUN_ID, "num_items": num_items,
                   "emb_dim": emb_dim, "cat_cardinalities": cardinalities,
                   "cat_cols": config.CAT_COLS, "max_hist": config.MAX_HIST,
                   "hist_take": config.HIST_TAKE, "dropout": DROPOUT,
                   "weight_decay": WEIGHT_DECAY, "batch": BATCH,
                   "epochs": NUM_EPOCHS, "seed": config.SEED}, f, indent=2)

    tr_ds = TaobaoDS(tr, encoders)
    va_ds = TaobaoDS(va, encoders)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
    va_dl = DataLoader(va_ds, batch_size=BATCH * 2, shuffle=False, collate_fn=collate)

    model = DIN(num_items, emb_dim, cardinalities, dropout=DROPOUT).to(DEVICE)
    # 用 SCL 向量预热物品 embedding（仅初始化一次，之后随梯度更新）
    with torch.no_grad():
        model.item_emb.weight.copy_(torch.from_numpy(emb_init))

    # 类不平衡加权：正样本仅约 13.7%，用 neg/pos 作 pos_weight 提升 AUC 与校准
    pos = float(y[tr_idx].sum())
    neg = float(len(tr_idx) - pos)
    pos_weight = torch.tensor([neg / pos], device=DEVICE)
    # 大 embedding 用稍小 lr + 权重衰减（默认 1e-4，第三轮加强以对抗过拟合）
    # BCEWithLogitsLoss 支持 pos_weight（forward 已改为输出原始 logits）
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    # 验证 AUC 不再上升时自动把 lr 减半，帮助后期收敛、抑制过拟合
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3, min_lr=1e-5)

    history, interrupted = [], False
    best_auc, wait, PATIENCE = 0.0, 0, 5
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        ep_start = time.time()
        try:
            model.train()
            running, n = 0.0, 0
            for tgt, hist, lmask, hlen, cidx, yy in tr_dl:
                tgt, hist, lmask, hlen, cidx, yy = (
                    tgt.to(DEVICE), hist.to(DEVICE), lmask.to(DEVICE),
                    hlen.to(DEVICE), cidx.to(DEVICE), yy.to(DEVICE),
                )
                opt.zero_grad()
                loss = loss_fn(model(tgt, hist, lmask, hlen, cidx), yy)
                loss.backward()
                opt.step()
                running += loss.item() * len(yy)
                n += len(yy)
            # 验证 AUC
            model.eval()
            preds, gts = [], []
            with torch.no_grad():
                for tgt, hist, lmask, hlen, cidx, yy in va_dl:
                    tgt, hist, lmask, hlen, cidx = (
                        tgt.to(DEVICE), hist.to(DEVICE), lmask.to(DEVICE),
                        hlen.to(DEVICE), cidx.to(DEVICE),
                    )
                    preds.append(model(tgt, hist, lmask, hlen, cidx).cpu().numpy())
                    gts.append(yy.numpy())
            auc = roc_auc_score(np.concatenate(gts), np.concatenate(preds))
            train_loss = running / n
            scheduler.step(auc)
            history.append({"epoch": epoch, "train_loss": train_loss, "valid_auc": auc})

            torch.save(model.state_dict(), os.path.join(config.OUT_DIR, f"epoch_{epoch + 1:03d}.pt"))
            with open(os.path.join(config.OUT_DIR, f"epoch_{epoch + 1:03d}.json"), "x") as f:
                json.dump(history[-1], f, indent=2)

            # 早停：best-AUC 连续 PATIENCE 轮无提升则停止，并保存最优权重
            if auc > best_auc + 1e-5:
                best_auc = auc
                wait = 0
                torch.save(model.state_dict(),
                           os.path.join(config.OUT_DIR, "best_din.pt"))
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"[early-stop] 验证 AUC 连续 {PATIENCE} 轮未提升，"
                          f"早停在 epoch {epoch}，best={best_auc:.4f}")
                    break

            ep_cost = time.time() - ep_start
            done = epoch + 1
            eta = ep_cost * (NUM_EPOCHS - done)
            print(f"[epoch {epoch}] train_loss={train_loss:.4f}  valid AUC={auc:.4f}"
                  f"  (本轮 {ep_cost/60:.1f} 分, 预计剩余 {eta/3600:.1f} 小时)")
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[stop] 检测到中断，已保存前 {len(history)} 个 epoch 的进度")
            break

    final_auc = best_auc if best_auc > 0 else (history[-1]["valid_auc"] if history else None)
    status = "中断/早停" if (interrupted or wait >= PATIENCE) else "完成"
    print(f"[done] DIN 在 {DEVICE} 上训完（{status}），valid AUC={final_auc}")
    # 以 prepare.py 落盘的「真实序列方向」为准
    _meta_path = os.path.join(config.PROCESSED_DIR, "prep_meta.json")
    _hist_take = config.HIST_TAKE
    _emb_vocab = None
    if os.path.exists(_meta_path):
        try:
            _m = json.load(open(_meta_path))
            _hist_take = _m.get("hist_take", config.HIST_TAKE)
            _emb_vocab = _m.get("emb_vocab")
        except Exception:
            pass
    with open(os.path.join(config.OUT_DIR, "din_metrics.json"), "w") as f:
        json.dump({"device": DEVICE, "hist_take": _hist_take,
                   "max_hist": config.MAX_HIST, "emb_trainable": True,
                   "emb_vocab": _emb_vocab, "epochs_planned": NUM_EPOCHS,
                   "epochs_done": len(history), "interrupted": interrupted,
                   "history": history, "valid_auc": float(final_auc) if final_auc else None},
                  f, indent=2, ensure_ascii=False)
    print(f"[done] 指标 -> outputs/din_metrics.json")


if __name__ == "__main__":
    main()
