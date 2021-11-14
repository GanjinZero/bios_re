from tqdm import tqdm
import ujson
import os


UMLS_PATH = "/media/sda1/GanjinZero/UMLSBert/umls"


def byLineReader(filename):
    with open(filename, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            yield line
            line = f.readline()
    return

def load_umls_rel():
    reader = byLineReader(os.path.join(UMLS_PATH, "MRREL.RRF"))
    rel_set = set()
    head_dict = {}
    tail_dict = {}
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
        str_rel = "\t".join([cui0, cui1, re, rel])
        if not str_rel in rel_set:
            rel_set.update([str_rel])
            if not cui0 in head_dict:
                head_dict[cui0] = {}
            if not cui1 in head_dict[cui0]:
                head_dict[cui0][cui1] = []
            head_dict[cui0][cui1].append("\t".join([re, rel]))
            if not cui1 in tail_dict:
                tail_dict[cui1] = {}
            if not cui0 in tail_dict[cui1]:
                tail_dict[cui1][cui0] = []
            tail_dict[cui1][cui0].append("\t".join([re, rel]))

    with open('rel_set.json', 'w') as f:
        ujson.dump(list(rel_set), f, indent=2)

    with open('head_dict.json', 'w') as f:
        ujson.dump(head_dict, f, indent=2)

    with open('tail_dict.json', 'w') as f:
        ujson.dump(tail_dict, f, indent=2)

load_umls_rel()
