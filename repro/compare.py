import json, sys, os

paths = sys.argv[1:]
rows = []
for p in paths:
    d = json.load(open(p))
    am = d["aggregate_metrics"]
    row = {
        "file": os.path.basename(p),
        "n": d["total_questions"],
        "usage": d.get("llm_usage"),
        "fail": d.get("llm_failures"),
    }
    for k in ["overall", "category_1", "category_2", "category_3", "category_4", "category_5"]:
        if k in am:
            row[k] = (round(am[k]["f1"]["mean"] * 100, 2), round(am[k]["bleu1"]["mean"] * 100, 2),
                      int(am[k]["f1"]["count"] or 0))
    rows.append(row)

for r in rows:
    print("=" * 80)
    print(r["file"], "| questions:", r["n"])
    print("  usage:", r["usage"])
    print("  failures:", r["fail"])
    for k in ["overall", "category_1", "category_2", "category_3", "category_4", "category_5"]:
        if k in r:
            f1, b1, n = r[k]
            print(f"  {k:12s} F1 {f1:6.2f}  BLEU-1 {b1:6.2f}  (n={n})")
