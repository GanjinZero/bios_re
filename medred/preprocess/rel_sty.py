import ujson
from load_umls import UMLS
import os
from tqdm import tqdm


UMLS_PATH = "/media/sda1/GanjinZero/UMLSBert/umls"
umls = UMLS(UMLS_PATH, only_load_dict=True)
umls.load_sty()


def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()
    return

reader = byLineReader(os.path.join(UMLS_PATH, "MRREL.RRF"))
rel2sty = {}
for line in tqdm(reader, ascii=True):
    l = line.strip().split("|")
    cui0 = l[0]
    re = l[3]
    cui1 = l[4]
    rel = l[7]
    source = l[10]
    if not source in ['MED-RT', 'RXNORM', 'SNOMEDCT_US']:
        continue
    if not rel or cui0 == cui1:
        continue
    if not f'{re}\t{rel}' in rel2sty:
        rel2sty[f'{re}\t{rel}'] = set()
    if cui0 in umls.cui2sty and cui1 in umls.cui2sty:    
        rel2sty[f'{re}\t{rel}'].update([f'{umls.cui2sty[cui0]}\t{umls.cui2sty[cui1]}'])

rel2sty = {key:list(value) for key, value in rel2sty.items()}

with open('rel2sty.json', 'w') as f:
    ujson.dump(rel2sty, f, indent=2)
