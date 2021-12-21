# -*- coding: UTF-8 -*-
import json
import time
import copy
from tqdm import tqdm
import itertools
import re
import numpy as np

ds_39g = {
    "raw_text_path": "/platform_tech/aigraph/data/39g_data/pubmed_abstract_title_0.25_fulltext.txt",
    "tagged_text_path": "/platform_tech/aigraph/data/39g_data/pubmed_abstract_title_0.25_fulltext_bios_1029_tagged.txt",   
    "predict_path": "",
    "dataset_dir": "",
    "df_path": ""	
}

def read():

    term = {}
    str_cid = {} #str:cid

    # cid tid str
    for idx, line in enumerate(open("/platform_tech/wikidata/Concepts 2021-11-05.txt", "r", encoding = "utf-8")):
        if idx == 0:
            continue
        one = line.strip().split("\t")
        str_cid[one[2]] = one[0]
        term[one[2]] = 1

    cnt = 0
    cid_pair = {}
    tri = {}
    
    #"Entity1", "Qid1", "Relation", "Entity2", "Qid2"
    
    for idx, line in enumerate(tqdm(open("/platform_tech/wikidata/wiki_sentence_1125/triplets_1119.txt", "r", encoding = "utf-8"))):
        if idx == 0:
            continue
        t_t = line.strip().split("\t")
        
        if t_t[0] not in str_cid.keys() or t_t[3] not in str_cid.keys():
            continue
        
        if (t_t[0], t_t[3]) not in tri.keys():
            tri[(t_t[0], t_t[3])] = [[t_t[2],(t_t[1], t_t[4])]]
        else:
            tri[(t_t[0], t_t[3])] = tri[(t_t[0], t_t[3])] + [[t_t[2],(t_t[1], t_t[4])]]
            
        if (t_t[3], t_t[0]) not in tri.keys():
            tri[(t_t[3], t_t[0])] = [[t_t[2],(t_t[4], t_t[1])]]
        else:
            tri[(t_t[3], t_t[0])] = tri[(t_t[3], t_t[0])] + [[t_t[2],(t_t[4], t_t[1])]]

        cui_en1 = str_cid[t_t[0]] 
        cui_en2 = str_cid[t_t[3]]
        
        if cui_en1 not in cid_pair.keys():
            cid_pair[cui_en1] = [(t_t[2],), (cui_en2,)]
        else:
            cid_pair[cui_en1][0] = cid_pair[cui_en1][0] + (t_t[2], )
            cid_pair[cui_en1][1] = cid_pair[cui_en1][1] + (cui_en2, )

        if cui_en2 not in cid_pair.keys():
            cid_pair[cui_en2] = [(t_t[2],), (cui_en1,)]
        else:
            cid_pair[cui_en2][0] = cid_pair[cui_en2][0] + (t_t[2], )
            cid_pair[cui_en2][1] = cid_pair[cui_en2][1] + (cui_en1, )

####begin matching 
    print("################begin matching##################")
    writer = open('/platform_tech/yuanzheng/bios_re/data/1202/raw.json', 'w', encoding="utf-8") 
    
    with open(ds_39g["raw_text_path"], 'r', encoding = 'utf-8') as fp1, open(ds_39g["tagged_text_path"], 'r', encoding = 'utf-8') as fp2:
        for l1 in tqdm(fp1):
            sen = l1.strip()
            line2 = fp2.readline()
            tag = json.loads(line2.strip())
            if len(tag) <= 1:
                continue
            dic = {}
            sen_list = sen.split(". ")
            
            for one_sen in sen_list:
                left_idx = sen.index(one_sen)
                right_idx = left_idx + len(one_sen)
                need_tag = get_phrase(left_idx, right_idx, tag)
                if len(need_tag) < 2:
                    continue      
                for i in range(len(need_tag)):
                
                    x = need_tag[i]
                    if x not in str_cid.keys():
                        continue
                    x_id = str_cid[x]
                    if x_id not in cid_pair.keys():
                        continue
                        
                    #for y in need_tag[i+1:]:
                    for y in need_tag:
                        if y not in str_cid.keys():
                            continue
                        y_id = str_cid[y]
                        if y_id not in cid_pair[x_id][1]:
                            continue
                        if y_id == x_id:
                            continue
                            
                        dic['text'] = one_sen+"."
                        x_word = x
                        y_word = y
                        if (x_word, y_word) in tri.keys():
                            dic['relation'] = [rel[0] for rel in tri[(x_word,y_word)]]
                            dic["Qid_pairs"] = [q[1] for q in tri[(x_word,y_word)]]
                        else:
                            rel_id = [i for i in np.where(np.array(cid_pair[x_id][1]) == y_id)[0]]
                            dic['relation'] = list(set([cid_pair[x_id][0][x] for x in rel_id]))
                            dic["Qid_pairs"] = ["NaN"]
                        xidx_start = one_sen.find(x)
                        xidx_end = xidx_start + len(x)
                        yidx_start = one_sen.find(y)
                        yidx_end = yidx_start + len(y)
                        if xidx_start == -1 or yidx_start == -1:
                            continue
                        dic["h"] = {"pos": [xidx_start, xidx_end],"id": x_id, "name": x}                            
                        dic["t"] = {"pos": [yidx_start, yidx_end],"id": y_id, "name": y}
                        dic["distance"] = len(one_sen[xidx_end : yidx_start].split(" "))
                        writer.write(json.dumps(dic))
                        writer.write("\n")
                        cnt += 1
                        break
                    else: 
                        continue
                    break

    writer.close()
    return cnt
    
def get_phrase(left_idx, right_idx, tag):
    tagged_words = []
    for x in tag:
        if x["begin"] >= left_idx and x["end"] <= right_idx :
            tagged_words.append(x["phrase"])
        else:
            continue
    return tagged_words

if __name__ == "__main__":
    start_time = time.time()
    output_n = read()
    print('extracted {} sentences'.format(output_n))
    end_time = time.time()
    print("running time %s seconds"%(end_time-start_time))

