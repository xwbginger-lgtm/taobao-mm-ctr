"""
基线 CTR 模型（不使用行为序列，只看「用户属性 + 候选商品属性 + 统计特征」）。

特征（全部在 train 划分上 fit，避免泄漏）：
  - item_ctr : 候选商品的历史点击率（带平滑）
  - user_ctr : 用户的历史点击率（带平滑）
  - hist_len : 用户历史行为长度
  - 用户年龄/性别/省/市/城市等级 + 候选商品类目/市/省 的标签编码
对比基准：全局平均 CTR（盲猜）。最终看 LR / LightGBM 的 AUC 提升。

用法：  python train_baseline.py
输出：  outputs/metrics.json
"""
import json
import os

import config
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False

np.random.seed(config.SEED)
ALPHA = 10  # 点击率平滑


def main():
    config.prepare_run_dir("baseline")
    path = os.path.join(config.PROCESSED_DIR, "train_prepared.parquet")
    if not os.path.exists(path):
        raise SystemExit("先跑 python prepare.py")
    df = pd.read_parquet(path)
    y = df["click"].astype(int).values

    tr_idx, va_idx = train_test_split(
        np.arange(len(df)), test_size=0.1, random_state=config.SEED, stratify=y)
    tr, va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    # 统计特征只在 train 上算
    item_ctr = (tr.groupby(config.COL["target_item"])["click"].mean())
    user_ctr = (tr.groupby(config.COL["user_id"])["click"].mean())
    tr["item_ctr"] = tr[config.COL["target_item"]].map(item_ctr).fillna(item_ctr.mean())
    va["item_ctr"] = va[config.COL["target_item"]].map(item_ctr).fillna(item_ctr.mean())
    tr["user_ctr"] = tr[config.COL["user_id"]].map(user_ctr).fillna(user_ctr.mean())
    va["user_ctr"] = va[config.COL["user_id"]].map(user_ctr).fillna(user_ctr.mean())

    dense_feats = ["item_ctr", "user_ctr", "hist_len"]
    cat_feats = []
    encoders = {}
    for c in config.CAT_COLS:
        le = LabelEncoder()
        le.fit(tr[c].astype(str))
        # 用 map+fillna 代替 le.transform：验证集里的长尾类别（训练集没见过）
        # 统一填 0，避免 sklearn 遇到 unseen label 直接报错
        mapping = {v: i for i, v in enumerate(le.classes_)}
        encoders[c] = mapping
        tr[c + "_c"] = tr[c].astype(str).map(mapping).fillna(0).astype(int)
        va[c + "_c"] = va[c].astype(str).map(mapping).fillna(0).astype(int)
        cat_feats.append(c + "_c")
    feats = dense_feats + cat_feats

    # 用 DataFrame 承载，保留列名 -> LightGBM 才能正确识别「类别列」
    Xtr = pd.concat([tr[dense_feats], tr[cat_feats]], axis=1)[feats]
    Xva = pd.concat([va[dense_feats], va[cat_feats]], axis=1)[feats]

    base = y_tr.mean()
    auc0 = roc_auc_score(y_va, np.full_like(y_va, base, dtype=float))
    ll0 = log_loss(y_va, np.full_like(y_va, base, dtype=float))
    print(f"[data]  train={len(tr):,}  valid={len(va):,}  CTR={base:.4f}")
    print(f"[base]  盲猜            AUC={auc0:.4f}  LogLoss={ll0:.4f}")

    results = {"n_train": int(len(tr)), "n_valid": int(len(va)), "global_ctr": float(base),
               "baseline_mean_predictor": {"auc": float(auc0), "logloss": float(ll0)}}

    # LR：最简单的概率打分基线。
    # - 先用 StandardScaler 标准化（LR 对特征尺度敏感，不标准化 lbfgs 难收敛 -> 警告）
    # - max_iter 调大避免 lbfgs 不收敛警告
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr.values)
    Xtr_lr = scaler.transform(Xtr.values)
    Xva_lr = scaler.transform(Xva.values)
    lr = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", n_jobs=1)
    lr.fit(Xtr_lr, y_tr)
    p = lr.predict_proba(Xva_lr)[:, 1]
    results["logistic_regression"] = {"auc": float(roc_auc_score(y_va, p)),
                                      "logloss": float(log_loss(y_va, p))}
    print(f"[lr]    LogisticRegression  AUC={results['logistic_regression']['auc']:.4f}  "
          f"LogLoss={results['logistic_regression']['logloss']:.4f}")

    if HAVE_LGB:
        # LightGBM：梯度提升树。
        # - categorical_feature=cat_feats：明确告诉模型「这些列是类别」，
        #   否则它把整数编码当连续值去分裂（如「城市编码<157」），导致过拟合、LogLoss 爆炸。
        # - 去掉 class_weight + 加正则：让概率更校准，LogLoss 回到合理区间。
        gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 min_child_samples=200, reg_lambda=2.0,
                                 subsample=0.8, n_jobs=1,
                                 random_state=config.SEED, verbose=-1)
        gbm.fit(Xtr, y_tr, categorical_feature=cat_feats)
        p2 = gbm.predict_proba(Xva)[:, 1]
        results["lightgbm"] = {"auc": float(roc_auc_score(y_va, p2)),
                               "logloss": float(log_loss(y_va, p2))}
        print(f"[lgb]   LightGBM          AUC={results['lightgbm']['auc']:.4f}  "
              f"LogLoss={results['lightgbm']['logloss']:.4f}")

    import pickle
    with open(os.path.join(config.OUT_DIR, "baseline_bundle.pkl"), "xb") as f:
        pickle.dump({"lr": lr, "lightgbm": gbm if HAVE_LGB else None,
                     "scaler": scaler, "encoders": encoders, "features": feats,
                     "item_ctr": item_ctr, "user_ctr": user_ctr,
                     "global_ctr": float(base), "run_id": config.RUN_ID}, f)
    np.savez(os.path.join(config.OUT_DIR, "split.npz"), train=tr_idx, valid=va_idx)
    with open(os.path.join(config.OUT_DIR, "metrics.json"), "x") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] 指标 -> outputs/metrics.json")


if __name__ == "__main__":
    main()
