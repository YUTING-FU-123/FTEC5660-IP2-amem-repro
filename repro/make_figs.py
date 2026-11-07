"""Build the two report figures from the result JSONs."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figs")
os.makedirs(OUT, exist_ok=True)

CATS = [("category_1", "multi-hop"), ("category_2", "temporal"),
        ("category_3", "open-domain"), ("category_4", "single-hop")]
PAPER = {"multi-hop": 27.02, "temporal": 45.85, "open-domain": 12.14, "single-hop": 44.65}


def weighted(paths, split):
    num = den = 0.0
    for p in paths:
        m = json.load(open(p))["aggregate_metrics"].get(split)
        if not m:
            continue
        num += m["f1"]["mean"] * m["f1"]["count"]
        den += m["f1"]["count"]
    return (num / den * 100) if den else None


def collect(pattern):
    return {split: weighted(sorted(glob.glob(os.path.join(ROOT, "repro", "out", pattern))), split)
            for split, _ in CATS}


full = collect("conv*_full_deepseek-flash_k10_s42.json")
nome = collect("conv*_nome_deepseek-flash_k10_s42.json")

# --- figure 1: paper vs ours ------------------------------------------------
labels = [name for _, name in CATS]
ours = [full[s] for s, _ in CATS]
paper = [PAPER[name] for name in labels]

x = range(len(labels))
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w = 0.38
ax.bar([i - w / 2 for i in x], paper, w, label="paper (GPT-4o-mini)", color="#9aa5b1")
ax.bar([i + w / 2 for i in x], ours, w, label="ours (DeepSeek V4.1 Flash)", color="#2f6f9f")
for i, (p, o) in enumerate(zip(paper, ours)):
    ax.text(i - w / 2, p + 0.6, f"{p:.1f}", ha="center", fontsize=8, color="#333")
    ax.text(i + w / 2, o + 0.6, f"{o:.1f}", ha="center", fontsize=8, color="#1b4a6b")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("token-level F1 (%)")
ax.set_title("A-Mem on LoCoMo: reported vs reproduced (3 conversations, 500 questions)")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_reproduction.png"), dpi=200)
plt.close(fig)

# --- figure 2: ablation ----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                               gridspec_kw={"width_ratios": [2.1, 1]})
b = [full[s] for s, _ in CATS]
n = [nome[s] for s, _ in CATS]
ax1.bar([i - w / 2 for i in x], b, w, label="baseline", color="#2f6f9f")
ax1.bar([i + w / 2 for i in x], n, w, label="w/o memory evolution", color="#c98b3a")
for i, (bv, nv) in enumerate(zip(b, n)):
    ax1.text(i - w / 2, bv + 0.6, f"{bv:.1f}", ha="center", fontsize=8)
    ax1.text(i + w / 2, nv + 0.6, f"{nv:.1f}", ha="center", fontsize=8)
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels)
ax1.set_ylabel("token-level F1 (%)")
ax1.set_title("Ablation: removing memory evolution")
ax1.legend(frameon=False, fontsize=9)
ax1.spines[["top", "right"]].set_visible(False)

tokens = [9.96, 6.25]
ax2.bar(["baseline", "w/o ME"], tokens, color=["#2f6f9f", "#c98b3a"], width=0.55)
for i, t in enumerate(tokens):
    ax2.text(i, t + 0.2, f"{t:.2f}M", ha="center", fontsize=9)
ax2.set_ylabel("prompt tokens (millions)")
ax2.set_title("Cost of 500 questions")
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_ablation.png"), dpi=200)
plt.close(fig)

print("wrote", OUT)
for s, name in CATS:
    print(f"{name:12s} paper {PAPER[name]:5.2f}  ours {full[s]:5.2f}  w/oME {nome[s]:5.2f}")
