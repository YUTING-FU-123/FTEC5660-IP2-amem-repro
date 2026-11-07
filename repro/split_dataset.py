#!/usr/bin/env python3
"""Split data/locomo10.json into one file per conversation.

The evaluation harness samples conversations with `--ratio`, which always takes
the FIRST N conversations.  Splitting lets us run any subset (and run several
conversations in parallel) without changing the harness.

Usage:
    python repro/split_dataset.py               # writes repro/datasets/conv_0.json .. conv_9.json
    python repro/split_dataset.py --indices 0 3 5
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "locomo10.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "repro", "datasets"))
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    args = ap.parse_args()

    with open(args.dataset) as f:
        samples = json.load(f)

    idx = args.indices if args.indices else list(range(len(samples)))
    os.makedirs(args.outdir, exist_ok=True)
    for i in idx:
        path = os.path.join(args.outdir, f"conv_{i}.json")
        with open(path, "w") as f:
            json.dump([samples[i]], f)
        n_qa = len(samples[i]["qa"])
        conv = samples[i]["conversation"]
        n_turns = sum(len(v) for k, v in conv.items()
                      if k.startswith("session_") and not k.endswith("date_time"))
        print(f"{path}: {n_turns} turns, {n_qa} QA")


if __name__ == "__main__":
    main()
