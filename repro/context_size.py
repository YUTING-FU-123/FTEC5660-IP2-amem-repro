"""Estimate the retrieved-context size per question from a harness log.

The harness logs 'Raw Context: <...>' for every question; we split on the
logger prefix so multi-line contexts stay intact.
"""
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="ignore").read()
# drop carriage-return progress bars
text = text.replace("\r", "\n")
chunks = re.split(r"(?:INFO:locomo_eval_robust:|INFO - )", text)
ctx, prompt = [], []
mode = None
for c in chunks:
    if c.startswith("Raw Context:"):
        mode = "ctx"
        buf = c[len("Raw Context:"):].strip()
    elif c.startswith("User Prompt:"):
        mode = "prompt"
        buf = c[len("User Prompt:"):].strip()
    elif c.startswith("Category:"):
        mode = None
        buf = c[len("Category:"):].strip()
        continue
    else:
        if mode == "ctx":
            ctx[-1] += "\n" + c.strip()
        elif mode == "prompt":
            prompt[-1] += "\n" + c.strip()
        continue
    (ctx if mode == "ctx" else prompt).append(buf)

def stats(name, xs):
    if not xs:
        print(f"{name}: none found")
        return
    chars = [len(x) for x in xs]
    chars.sort()
    mean = sum(chars) / len(chars)
    print(f"{name}: n={len(chars)} mean_chars={mean:.0f} (~{mean/4:.0f} tokens) "
          f"median={chars[len(chars)//2]} p90={chars[int(len(chars)*0.9)]}")

stats("raw context", ctx)
stats("full prompt", prompt)
