import json
d = json.load(open('data/locomo10.json'))
for i, s in enumerate(d):
    conv = s['conversation']
    turns = sum(len(v) for k, v in conv.items()
                if k[:8] == 'session_' and not k.endswith('date_time'))
    print(f"conv {i}: turns={turns:4d} qa={len(s['qa']):3d} sample_id={s['sample_id']}")
