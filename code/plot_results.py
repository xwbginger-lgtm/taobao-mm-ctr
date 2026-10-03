# -*- coding: utf-8 -*-
"""
Taobao-MM 实验可视化
一次生成 8 张图，输出到 E:/taobao-mm/figures/
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ============ 中文字体配置 ============
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT_DIR = "E:/taobao-mm/figures"
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    plt.close(fig)


# ============ 图 1：数据量饱和曲线 ============
shards = [1, 2, 4]
rows = [500_000, 1_000_000, 2_000_000]
din_auc = [0.6381, 0.6436, 0.6426]
lr_auc = [0.6207, 0.6232, 0.6235]

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(shards, din_auc, marker="o", color="steelblue", linewidth=2,
         markersize=8, label="DIN")
ax1.plot(shards, lr_auc, marker="s", color="orange", linewidth=2,
         markersize=8, label="LogisticRegression")
for x, y in zip(shards, din_auc):
    ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                 xytext=(0, 10), ha="center", color="steelblue", fontsize=9)
ax1.set_xlabel("分片数", fontsize=11)
ax1.set_ylabel("验证集 AUC", fontsize=11)
ax1.set_xticks(shards)
ax1.set_ylim(0.61, 0.65)
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right")
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(shards)
ax2.set_xticklabels([f"{r//10000}万" for r in rows])
ax2.set_xlabel("采样行数", fontsize=11)
ax1.set_title("数据量饱和曲线：AUC vs 分片数", fontsize=12)
save(fig, "图1_数据量饱和曲线.png")

# ============ 图 2：模型对比柱状图 ============
models = ["随机", "LightGBM", "LogisticRegression", "DIN"]
aucs = [0.5000, 0.5792, 0.6235, 0.6426]
colors = ["gray", "orange", "steelblue", "green"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, aucs, color=colors, width=0.6)
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{auc:.4f}", ha="center", fontsize=10)
ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
ax.set_ylabel("验证集 AUC", fontsize=11)
ax.set_ylim(0.45, 0.68)
ax.set_title("模型对比（4 分片、200 万行）", fontsize=12)
ax.grid(axis="y", alpha=0.3)
save(fig, "图2_模型对比柱状图.png")

# ============ 图 3：AUC 收敛曲线 ============
epochs = [0, 1, 2, 3, 4, 5]
r009 = [0.5532, 0.5671, 0.6217, 0.6347, 0.6366, 0.6381]
r010 = [0.5663, 0.5889, 0.6345, 0.6413, 0.6419, 0.6436]
r008 = [0.5876, 0.5931, 0.6292, 0.6399, 0.6426, 0.6416]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, r009, marker="o", label="1 分片, 50 万行")
ax.plot(epochs, r010, marker="s", label="2 分片, 100 万行")
ax.plot(epochs, r008, marker="^", label="4 分片, 200 万行")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("验证集 AUC", fontsize=11)
ax.set_xticks(epochs)
ax.set_ylim(0.54, 0.66)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
ax.set_title("DIN 收敛曲线", fontsize=12)
save(fig, "图3_AUC收敛曲线.png")

# ============ 图 4：weight_decay 对比 ============
fig, ax = plt.subplots(figsize=(7, 5))
configs = ["wd=1e-3", "wd=1e-4"]
vals = [0.5882, 0.6436]
bars = ax.bar(configs, vals, color=["#c0504d", "#4f81bd"], width=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{v:.4f}", ha="center", fontsize=11)
ax.set_ylabel("验证集 AUC", fontsize=11)
ax.set_ylim(0.55, 0.68)
ax.set_title("weight_decay 对 AUC 的影响（2 分片、100 万行、20 万词表）", fontsize=12)
ax.grid(axis="y", alpha=0.3)
ax.annotate("", xy=(1, 0.6436), xytext=(0, 0.5882),
            arrowprops=dict(arrowstyle="->", color="green", lw=2))
ax.text(0.5, 0.62, "+0.0554", ha="center", color="green", fontsize=11, fontweight="bold")
save(fig, "图4_weight_decay对比.png")

# ============ 图 5：词表大小 vs AUC / 训练时间 ============
fig, ax1 = plt.subplots(figsize=(8, 5))
vocab_labels = ["200 万词表", "20 万词表"]
vocab_auc = [0.6438, 0.6436]
vocab_time = [22.0, 3.0]

x = np.arange(len(vocab_labels))
width = 0.35

bars1 = ax1.bar(x - width/2, vocab_auc, width, label="验证集 AUC", color="steelblue")
ax1.set_ylabel("验证集 AUC", color="steelblue", fontsize=11)
ax1.set_ylim(0.55, 0.68)
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_xticks(x)
ax1.set_xticklabels(vocab_labels)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, vocab_time, width, label="单 epoch 耗时(分钟)", color="orange")
ax2.set_ylabel("单 epoch 耗时 (分钟)", color="orange", fontsize=11)
ax2.tick_params(axis="y", labelcolor="orange")

for bar, v in zip(bars1, vocab_auc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f"{v:.4f}", ha="center", fontsize=10, color="steelblue")
for bar, v in zip(bars2, vocab_time):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v:.1f} 分", ha="center", fontsize=10, color="orange")

ax1.set_title("词表大小对 AUC 和训练时间的影响（2 分片、100 万行、wd=1e-4）", fontsize=12)
save(fig, "图5_词表大小对比.png")

# ============ 图 6：HIST_TAKE 对比 ============
fig, ax = plt.subplots(figsize=(7, 5))
groups = ["2 分片, 50 万行\nwd=1e-3", "2 分片, 50 万行\nwd=1e-3"]
first_vals = [0.5882, 0.5495]
last_vals = [0.5881, 0.5372]
x = np.arange(len(groups))
width = 0.35

bars1 = ax.bar(x - width/2, first_vals, width, label="HIST_TAKE=first", color="steelblue")
bars2 = ax.bar(x + width/2, last_vals, width, label="HIST_TAKE=last", color="orange")
for bars, vals in [(bars1, first_vals), (bars2, last_vals)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.4f}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(["20 万词表", "200 万词表"])
ax.set_ylabel("验证集 AUC", fontsize=11)
ax.set_ylim(0.50, 0.62)
ax.set_title("序列方向 first vs last 对比", fontsize=12)
ax.legend()
ax.grid(axis="y", alpha=0.3)
save(fig, "图6_序列方向对比.png")

# ============ 图 7：训练 loss 曲线 ============
r009_loss = [1.1881, 1.1747, 1.1395, 1.0751, 1.0173, 0.9774]
r010_loss = [1.1864, 1.1705, 1.1399, 1.0806, 1.0450, 1.0241]
r008_loss = [1.1806, 1.1664, 1.1526, 1.1173, 1.0955, 1.0832]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, r009_loss, marker="o", label="1 分片, 50 万行")
ax.plot(epochs, r010_loss, marker="s", label="2 分片, 100 万行")
ax.plot(epochs, r008_loss, marker="^", label="4 分片, 200 万行")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("训练集 loss", fontsize=11)
ax.set_xticks(epochs)
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("训练 loss 曲线", fontsize=12)
save(fig, "图7_训练loss曲线.png")

# ============ 图 8：LR 基线随数据量变化 ============
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(shards, lr_auc, marker="s", color="orange", linewidth=2, markersize=8)
for x, y in zip(shards, lr_auc):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                xytext=(0, 10), ha="center", color="orange", fontsize=10)
ax.set_xlabel("分片数", fontsize=11)
ax.set_ylabel("验证集 AUC", fontsize=11)
ax.set_xticks(shards)
ax.set_ylim(0.61, 0.63)
ax.grid(alpha=0.3)
ax.set_title("LR 基线随数据量的变化", fontsize=12)
save(fig, "图8_LR随数据量变化.png")

print("\n全部完成。图片在:", OUT_DIR)
