import ujson


with open('count.json', 'r') as f:
    count = ujson.load(f)
with open('inverse_rel.json', 'r') as f:
    inverse_rel = ujson.load(f)
with open('head_dict.json', 'r') as f:
    head_dict = ujson.load(f)

use_rel = set()
for cui0 in head_dict:
    for cui1 in head_dict[cui0]:
        for rel in head_dict[cui0][cui1]:
            use_rel.update([rel])

# use_rel = set()
# for key in count:
#     if key in ['MED-RT', 'RXNORM', 'SNOMEDCT_US']:
#         use_rel.update(count[key])

use_rel = [rel for rel in use_rel if rel.split('\t')[1] not in inverse_rel]
use_rel.sort()
use_rel = ['NA'] + use_rel

with open('rel2id.json', 'w') as f:
    ujson.dump({rel:i for i, rel in enumerate(use_rel)}, f, indent=2)
