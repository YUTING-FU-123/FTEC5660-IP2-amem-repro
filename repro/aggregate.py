"""Aggregate per-conversation result files into one table (question-weighted).

Usage:
    python repro/aggregate.py repro/out/conv*_full*.json
    python repro/aggregate.py --arm nome repro/out/*.json
"""
import argparse
import glob
import json
import os

CATS = {"category_1": "multi-hop", "category_2": "temporal", "category_3": "open-domain",
        "category_4": "single-hop", "category_5": "adversarial"}


def load(paths):
    out = {}
    for p in paths:
        d = json.load(open(p))
        for split, m in d["aggregate_metrics"].items():
            f1, b1 = m["f1"], m["bleu1"]
            out.setdefault(split, []).append(
                (f1["mean"], f1["count"], b1["mean"], d.get("llm_usage", {}), d.get("llm_failures", {}))
            )
    return out


def weighted(rows):
    n = sum(r[1] for r in rows)
    f1 = sum(r[0] * r[1] for r in rows) / n
    b1 = sum(r[2] * r[1] for r in rows) / n
    return f1, b1, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    paths = []
    for f in args.files:
        paths.extend(glob.glob(f))

    data = load(sorted(paths))
    print(f"# {args.label}  ({len(paths)} runs)")
    print(f"{'split':12s} {'F1':>7s} {'BLEU-1':>7s} {'n':>5s}")
    for split in ["overall"] + list(CATS):
        if split not in data:
            continue
        f1, b1, n = weighted(data[split])
        name = CATS.get(split, split)
        print(f"{name:12s} {f1*100:7.2f} {b1*100:7.2f} {n:5d}")

    calls = sum(u.get("calls", 0) for rows in data.values() for _, _, _, u, _ in rows) // max(1, len(data))
    prompt = sum(u.get("prompt_tokens", 0) for rows in data.values() for _, _, _, u, _ in rows) // max(1, len(data))
    comp = sum(u.get("completion_tokens", 0) for rows in data.values() for _, _, _, u, _ in rows) // max(1, len(data))
    print(f"\nLLM calls: {calls}  prompt tokens: {prompt:,}  completion tokens: {comp:,}")
    fails = {}
    for rows in data.values():
        for _, _, _, _, f in rows:
            for k, v in (f or {}).items():
                fails[k] = fails.get(k, 0) + v
    print("failures:", fails or "none")


if __name__ == "__main__":
    main()
