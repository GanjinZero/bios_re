import os
import json

res_path = '/platform_tech/yuanzheng/bios_re/output_1203/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_coder/all_predict.json'
with open(res_path, 'r') as f:
    lines = f.readlines()

dfs = [json.loads(line) for line in lines]

res = {}
for df in dfs:
    h = df['h']
    t = df['t']
    rel = df['predict_rel']

    if len(h.split()) < 2:
        continue
    if len(t.split()) < 2:
        continue

    if not f'{h}|||{t}' in res:
        res[f'{h}|||{t}'] = set()
    res[f'{h}|||{t}'].update(rel)

res = {key:list(value) for key, value in res.items()}

with open('summary.json', 'w') as f:
    json.dump(res, f, indent=2)
    
