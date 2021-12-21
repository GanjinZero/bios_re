import json
with open('./data/rel2sty.json', 'r') as f:
    rel2sty = json.load(f)

rel_dict = {}
rel_inverse_dict = {}
for _, value in rel2sty.items():
    for stys in value:
        t0, t1 = stys.split('|')
        if not t0 in rel_dict:
            rel_dict[t0] = set()
        if not t1 in rel_inverse_dict:
            rel_inverse_dict[t1] = set()
        rel_dict[t0].update([t1])
        rel_inverse_dict[t1].update([t0])

for key in rel_dict:
    rel_dict[key] = list(rel_dict[key])
for key in rel_inverse_dict:
    rel_inverse_dict[key] = list(rel_inverse_dict[key])

with open('./data/sty_pair.json', 'w') as f:
    json.dump({'forward':rel_dict, 'backward':rel_inverse_dict}, f, indent=2)
