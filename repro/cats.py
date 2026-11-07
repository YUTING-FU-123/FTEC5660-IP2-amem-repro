import json
d = json.load(open('data/locomo10.json'))
for i, s in enumerate(d):
    cats = {}
    for q in s['qa']:
        cats[q.get('category')] = cats.get(q.get('category'), 0) + 1
    conv = s['conversation']
    turns = sum(len(v) for k, v in conv.items()
                if k[:8] == 'session_' and not k.endswith('date_time'))
    print(f"conv {i}: turns={turns:4d} qa={len(s['qa']):3d} cats={dict(sorted(cats.items()))}")
