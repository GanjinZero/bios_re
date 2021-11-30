import os
from tqdm import tqdm
import json
from load_bios import BIOS


with open('data/triplets_1119.txt', 'r') as f:
    lines = f.readlines()[1:]

bios = BIOS('./bios')
string2type = {}
for term, string in tqdm(bios.term2string.items()):
    cui = bios.term2cui[term]
    sty = bios.cui2type[cui]
    string2type[string] = sty

rel2sty = {}
for line in lines:
    h, _, rel, t, _ = line.strip().split('\t') 
    if not rel in rel2sty:
        rel2sty[rel] = set()
    h_types = string2type.get(h, 'None')
    t_types = string2type.get(t, 'None')
    if h_types != "None" and t_types != "None":
        for h_type in h_types:
            for t_type in t_types:
                rel2sty[rel].update(['|'.join([h_type, t_type])])

for rel in rel2sty:
    rel2sty[rel] = list(rel2sty[rel])

with open('data/rel2sty.json', 'w') as f:
    json.dump(rel2sty, f, indent=2)
