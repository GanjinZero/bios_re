from load_umls import UMLS


umls_path = '../../umls'
umls = UMLS(umls_path, only_load_dict=True)
umls.load_sty()

import ujson
with open('cui2sty.json', 'w') as f:
    ujson.dump(umls.cui2sty, f, indent=2)

with open('sty2id.json',  'w') as f:
    ujson.dump({sty:idx for idx, sty in enumerate(set(umls.cui2sty.values()))}, f, indent=2)

