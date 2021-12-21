# -*- coding: UTF-8 -*-
import json
import time
import copy
from tqdm import tqdm
import itertools
import re
import numpy as np
from random import sample, random
from load_bios import BIOS

ds_39g = {
    "bios": '/platform_tech/wikidata/',
    "raw_text_path": "/platform_tech/aigraph/data/39g_data/pubmed_abstract_title_0.25_fulltext.txt",
    "tagged_text_path": "/platform_tech/aigraph/data/39g_data/pubmed_abstract_title_0.25_fulltext_bios_1029_tagged.txt", 
    "triplets_path": "/platform_tech/wikidata/wiki_sentence_1125/triplets_1119.txt",
    "sty_pair_path": "/platform_tech/yuanzheng/bios_re/data/sty_pair.json",
    "predict_path": "/platform_tech/yuanzheng/bios_re/data/1221/dataset.json"
}

bios = BIOS(ds_39g['bios'])
with open(ds_39g['sty_pair_path'], 'r') as f:
    sty_pair = json.load(f)

def get_phrase(left_idx, right_idx, tag):
    tagged_words = []
    for x in tag:
        if x["begin"] >= left_idx and x["end"] <= right_idx :
            tagged_words.append(x["phrase"])
        else:
            continue
    return tagged_words

def get_cui(ent):
    term_id = bios.string2term.get(ent, 'NNN')
    cui_id = bios.term2cui.get(term_id, 'NNN')
    return cui_id

def read():

    #"Entity1", "Qid1", "Relation", "Entity2", "Qid2"
    tri = {}
    use_cuis = set()
    for idx, line in enumerate(tqdm(open(ds_39g["triplets_path"], "r", encoding = "utf-8"))):
        if idx == 0:
            continue
        ent1, qid1, rel, ent2, qid2 = line.strip().split("\t")
       
        cui1 = get_cui(ent1)
        cui2 = get_cui(ent2)
        if cui1 == 'NNN' or cui2 == 'NNN':
            continue

        use_cuis.update([cui1, cui2])

        if not (cui1, cui2) in tri:
            tri[(cui1, cui2)] = set()
        tri[(cui1, cui2)].update([rel])

        if not (cui2, cui1) in tri:
            tri[(cui2, cui1)] = set()
        tri[(cui2, cui1)].update(["inverse " + rel])
        
    print("################begin matching##################")
    writer = open(ds_39g['predict_path'], 'w', encoding="utf-8") 
    
    cnt = 0
    neg = 0
    with open(ds_39g["raw_text_path"], 'r', encoding = 'utf-8') as fp1, open(ds_39g["tagged_text_path"], 'r', encoding = 'utf-8') as fp2:
        for idx, l1 in tqdm(enumerate(fp1)):
            # for debug
            """
            if idx == 10000:
                print(cnt, neg)
                import sys
                sys.exit()
            """
            if idx % 200000 == 0:
                print(idx, cnt, neg)

            sen = l1.strip()
            tag = json.loads(fp2.readline().strip())

            if len(tag) <= 1:
                continue

            sen_list = sen.split(". ")
            
            for one_sen in sen_list:
                u = random()

                left_idx = sen.index(one_sen)
                right_idx = left_idx + len(one_sen)
                need_tag = get_phrase(left_idx, right_idx, tag)
                need_tag_cui = {}
                need_tag_idx = {}
                for x in need_tag:
                    cui_x = get_cui(x)
                    if cui_x == 'NNN':
                        continue
                    if cui_x in use_cuis:
                        x_idx = one_sen.find(x)
                        if x_idx >= 0:
                            need_tag_cui[x] = cui_x
                            need_tag_idx[x] = x_idx
                
                if len(need_tag_cui) < 2:
                    continue

                permit_neg_flag = True
                output_flag = False
                dic = {}
                dic['text'] = one_sen + "."


                for x in need_tag_cui:
                    cui_x = need_tag_cui[x]
                    sty_x = bios.cui2type[cui_x][0]
                    for y in need_tag_cui:
                        
                        cui_y = need_tag_cui[y]
                        if cui_x == cui_y:
                            continue
                            
                        if (cui_x, cui_y) not in tri:
                            if u < 0.1:
                                if permit_neg_flag:                                
                                    sty_y = bios.cui2type[cui_y][0]
                                    if sty_y not in sty_pair['forward'][sty_x] and sty_y not in sty_pair['backward'][sty_x]:
                                        continue
                                    dic['relation'] = ['NA']
                                    permit_neg_flag = False
                                else:
                                    continue
                            else:
                                continue
                        else:
                            dic['relation'] = list(tri[(cui_x, cui_y)])

                        xidx_start = need_tag_idx[x]
                        xidx_end = xidx_start + len(x)
                        yidx_start = need_tag_idx[y]
                        yidx_end = yidx_start + len(y)

                        dic["h"] = {"pos": [xidx_start, xidx_end],"id": cui_x, "name": x}                            
                        dic["t"] = {"pos": [yidx_start, yidx_end],"id": cui_y, "name": y}
                        dic["distance"] = len(one_sen[xidx_end : yidx_start].split(" "))
                        writer.write(json.dumps(dic))
                        writer.write("\n")
                        
                        if dic['relation'] == ['NA']:
                            neg += 1
                        else:
                            cnt += 1

    writer.close()
    return cnt, neg
    

if __name__ == "__main__":
    start_time = time.time()
    output_n, neg_n = read()
    print('extracted {} sentences'.format(output_n))
    end_time = time.time()
    print("running time %s seconds"%(end_time-start_time))

