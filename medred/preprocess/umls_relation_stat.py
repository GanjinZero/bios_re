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
    count = {}

    for line in tqdm(reader, ascii=True):
        l = line.strip().split("|")
        cui0 = l[0]
        re = l[3]
        cui1 = l[4]
        rel = l[7]
        if not rel or cui0 == cui1:
            continue
        str_rel = "\t".join([cui0, cui1, re, rel])
        source = l[10]
        if not source in count:
            count[source] = {}
        if not rel in count[source]:
            count[source][rel] = 0
        count[source][rel] += 1

    with open('count.json', 'w') as f:
        ujson.dump(count, f, indent=2)


load_umls_rel()
