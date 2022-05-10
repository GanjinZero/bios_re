# -*- coding: UTF-8 -*-
import json
import time
import copy
from tqdm import tqdm
import itertools
import re


ds_39g = {
    "raw_text_path": "/platform_tech/aigraph/data/39g_data/pubmed_abstract_title_0.25_fulltext.txt",
    "tagged_text_path": "/platform_tech/aigraph/data/39g_data/pubmed_abstract_title_0.25_fulltext_bios_1029_tagged.txt",   
    "predict_path": "",
    "dataset_dir": "",
    "df_path": ""	
}

def get_phrase(left_idx, right_idx, tag):
    tagged_words = []
    for x in tag:
        if x["begin"] >= left_idx and x["end"] <= right_idx :
            tagged_words.append(x["phrase"])
        else:
            continue
    return tagged_words

def read():

    #cid	sty
    cid_sty = {} #cid : sty
    for idx, line in enumerate(open("/platform_tech/wikidata/Semtypes 2021-11-05.txt", "r", encoding = "utf-8")):
        if idx == 0:
            continue
        tem = line.strip().split("\t")
        cid_sty[tem[0]] = tem[1]

    term = {}
    str_cid = {} #str:cid
    sty_pair = {}
    str_sty = {}
    
    # cid tid str
    for idx, line in enumerate(open("/platform_tech/wikidata/Concepts 2021-11-05.txt", "r", encoding = "utf-8")):
        if idx == 0:
            continue
        one = line.strip().split("\t")
        str_cid[one[2]] = one[0]
        term[one[2]] = 1
        str_sty[one[2]] = cid_sty[one[0]]
        
    need_sty = {}
        
    def check(x_sty, y_sty):
        xs = x_sty.split('|')
        ys = y_sty.split('|')
        for x in xs:
            if not x in need_sty:
                continue
            for y in ys:
                if y in need_sty[x]:
                    return True
        for y in ys:
            if not y in need_sty:
                continue
            for x in xs:
                if x in need_sty[y]:
                    return True
        return False

    #"Entity1", "Qid1", "Relation", "Entity2", "Qid2"
    for idx, line in enumerate(tqdm(open("/platform_tech/wikidata/wiki_sentence_1125/triplets_1119.txt", "r", encoding = "utf-8"))):
        if idx == 0:
            continue
        t_t = line.strip().split("\t")
        if t_t[0] not in term.keys() or t_t[3] not in term.keys():
            continue
            
        sty_en1 = str_sty[t_t[0]]
        sty_en2 = str_sty[t_t[3]]
        for s1 in sty_en1.split('|'):
            for s2 in sty_en2.split('|'):
                if not s1 in need_sty:
                    need_sty[s1] = set()
                if not s2 in need_sty:
                    need_sty[s2] = set()
                need_sty[s1].update([s2])
                need_sty[s2].update([s1])
    
    cnt = 0
####begin matching 
  
    print("################begin matching##################")
    writer = open("./inference_0103.json", 'w', encoding="utf-8") 
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
                need_tag = [x for x in need_tag if x in term]
                if len(need_tag) < 2:
                    continue    
  
                for i in range(len(need_tag)):
                    x = need_tag[i]
                    x_id = str_cid[x]
                    x_sty = str_sty[x]
                        
                    for y in need_tag[i+1:]:
                        y_id = str_cid[y]
                        if y_id == x_id:
                            continue
                        y_sty = str_sty[y]
                        if not check(x_sty, y_sty):
                            continue

                        dic['text'] = one_sen+"."
                        xidx_start = one_sen.find(x)
                        xidx_end = xidx_start + len(x)
                        yidx_start = one_sen.find(y)
                        yidx_end = yidx_start + len(y)
                        if xidx_start == -1 or yidx_start == -1:
                            continue
                        dic["h"] = {"pos": [xidx_start, xidx_end],"id": x_id, "name": x}                            
                        dic["t"] = {"pos": [yidx_start, yidx_end],"id": y_id, "name": y}
                        dic["distance"] = len(one_sen[xidx_end : yidx_start].split(" "))
                        if dic['distance'] > 10 or dic['distance'] < 3:
                            continue
                        writer.write(json.dumps(dic))
                        writer.write("\n")
                        cnt += 1
                        #break
                    else:
                        continue
                        
                    break

    writer.close()
    return cnt

if __name__ == "__main__":
    start_time = time.time()
    output_n = read()
    print('extracted {} sentences'.format(output_n))
    end_time = time.time()
    print("running time %s seconds"%(end_time-start_time))

