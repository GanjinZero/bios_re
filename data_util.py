import re
from random import sample
import torch
import os
from torch.utils.data import Dataset
import sys
import pandas as pd
import numpy as np
import math
import json, ujson
from transformers import AutoTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


def pad(l, pad_id, pad_length):
    now_len = len(l)
    if now_len < pad_length:
        l = l + [pad_id] * (pad_length - now_len)
    return l[0:pad_length]


class REDataset(Dataset):
    def __init__(self, data_path, mode, bag_size, truncated_length, coder_truncated_length, bert_path, coder_path, 
                 reverse_train=False, limit_dis=None, debug=False):
        self.data_path = data_path
        self.mode = mode

        self.bag_size = bag_size
        self.truncated_length = truncated_length
        self.coder_truncated_length = coder_truncated_length

        self.bert_tok = AutoTokenizer.from_pretrained(bert_path)
        if coder_path:
            self.coder_tok = AutoTokenizer.from_pretrained(coder_path)
        else:
            self.coder_tok = self.bert_tok

        self.path = os.path.join(data_path, f'{mode}.txt')

        self.rel2id_path = os.path.join(data_path, 'rel2id.json')
        with open(self.rel2id_path, 'r') as f:
            self.rel2id = json.load(f)

        self.reverse_train = reverse_train
        if limit_dis is not None:
            self.limit_dis = [int(x) for x in limit_dis.split(',')]
        else:
            self.limit_dis = None

        self.load(self.path, debug)

    def load(self, path, debug=False):
        f = open(path)

        self.bag_scope = []
        self.name2id = {}
        self.bag_name = []
        self.facts = {}

        if self.reverse_train:
            reverse_cuis_set = set()
            reverse_cuis_line = []

        idx = -1
        for line in tqdm(f):
            if debug:
                if idx >= 100000:
                    break
            idx += 1
            line = line.rstrip()
            if len(line) > 0:
                df = eval(line)
                cuis = "|".join([df['h']['id'], df['t']['id']])
                dis = int(df['distance'])
                if self.limit_dis is not None:
                    if dis > self.limit_dis[1] or dis < self.limit_dis[0]:
                        continue
            if cuis not in self.name2id:
                self.name2id[cuis] = len(self.name2id)
                self.bag_scope.append([])
                self.bag_name.append(cuis)
            if len(self.bag_scope[self.name2id[cuis]]) < 10 * self.bag_size:
                self.bag_scope[self.name2id[cuis]].append(df)
            if df['relation'][0] != "NA" and self.reverse_train:
                reverse_cuis = "|".join([df['t']['id'], df['h']['id']])
                reverse_cuis_line.append(df)
                reverse_cuis_set.update([reverse_cuis])

        if self.reverse_train:
            set_bag_name = set(self.bag_name)
            for df in reverse_cuis_line:
                reverse_cuis = "|".join([df['t']['id'], df['h']['id']])
                if reverse_cuis in set_bag_name:
                    continue
                new_df = df.copy()
                new_df['h'] = df['t']
                new_df['t'] = df['h']
                new_df['relation'] = ["NA"]
                new_df["Qid_pairs"]: ["NaN"] 

                if reverse_cuis not in self.name2id:
                    self.name2id[reverse_cuis] = len(self.name2id)
                    self.bag_scope.append([])
                    self.bag_name.append(reverse_cuis)
                if len(self.bag_scope[self.name2id[reverse_cuis]]) < 10 * self.bag_size:
                    self.bag_scope[self.name2id[reverse_cuis]].append(new_df)

        f.close()

        self.len = len(self.bag_scope)
        
        return


    def __len__(self):
        return self.len


    def tokenize(self, item):

        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        sentence = item['text']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        sent0 = self.bert_tok.tokenize(sentence[:pos_min[0]])
        ent0 = self.bert_tok.tokenize(sentence[pos_min[0]:pos_min[1]])
        sent1 = self.bert_tok.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = self.bert_tok.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = self.bert_tok.tokenize(sentence[pos_max[1]:])

        # if self.mask_entity:
        #     ent0 = ['[unused4]'] if not rev else ['[unused5]']
        #     ent1 = ['[unused5]'] if not rev else ['[unused4]']
        # else:
        ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
        ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.truncated_length - 1, pos1)
        pos2 = min(self.truncated_length - 1, pos2)
        
        indexed_tokens = self.bert_tok.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        ent0_tokens = self.coder_tok.convert_tokens_to_ids(['[CLS]'] + ent0 + ['[SEP]'])
        ent0_len = len(ent0_tokens)
        ent1_tokens = self.coder_tok.convert_tokens_to_ids(['[CLS]'] + ent1 + ['[SEP]'])
        ent1_len = len(ent1_tokens)

        # Padding
        indexed_tokens = pad(indexed_tokens, 0, self.truncated_length)
        ent0_tokens = pad(ent0_tokens, 0, self.coder_truncated_length)
        ent1_tokens = pad(ent1_tokens, 0, self.coder_truncated_length)

        # Attention mask
        att_mask = pad([1] * avai_len, 0, self.truncated_length)
        att_ent0 = pad([1] * ent0_len, 0, self.coder_truncated_length)
        att_ent1 = pad([1] * ent1_len, 0, self.coder_truncated_length)
        return indexed_tokens, att_mask, pos1, pos2, \
            ent0_tokens, att_ent0, ent1_tokens, att_ent1


    def __getitem__(self, index):
        bag = self.bag_scope[index]
        if 0 < self.bag_size <= len(bag):
            bag = sample(bag, self.bag_size) 
        #rel = self.rel2id[bag[0]['relation']]

        bag_id = [index] * len(bag)

        label = [0 for _ in range(len(self.rel2id))]
        for rel in bag[0]['relation']:
            if rel in self.rel2id:
                label[self.rel2id[rel]] = 1
        labels = [label] * len(bag)
        #labels = [rel] * len(bag)

        input_ids = []
        attention_mask = []
        ent0_pos = []
        ent1_pos = []
        ent0_input_ids = []
        ent0_att = []
        ent1_input_ids = []
        ent1_att = []

        for item in bag:
            indexed_tokens, att_mask, pos1, pos2, \
            ent0_tokens, att_ent0, ent1_tokens, att_ent1 = self.tokenize(item)
            input_ids.append(indexed_tokens)
            attention_mask.append(att_mask)
            ent0_pos.append(pos1)
            ent1_pos.append(pos2)
            ent0_input_ids.append(ent0_tokens)
            ent0_att.append(att_ent0)
            ent1_input_ids.append(ent1_tokens)
            ent1_att.append(att_ent1)

        return bag_id, input_ids, attention_mask, ent0_pos, ent1_pos, \
               ent0_input_ids, ent0_att, ent1_input_ids, ent1_att, labels

def my_collate_fn(batch):
    type_count = len(batch[0])
    batch_size = sum([len(x[0]) for x in batch])
    output = ()
    for i in range(type_count):
        tmp = []
        for item in batch:
            tmp.extend(item[i])
        if len(tmp) <= batch_size:
            output += (torch.LongTensor(tmp),)
        elif isinstance(tmp[0], int):
            output += (torch.LongTensor(tmp).reshape(batch_size, -1),)
        elif isinstance(tmp[0], list):
            dim_y = len(tmp[0])
            output += (torch.LongTensor(tmp).reshape(batch_size, -1, dim_y),)
    return output


class PredictOneDataset(REDataset):
    def __init__(self, data_path, mode, bag_size, truncated_length, coder_truncated_length, bert_path, coder_path, debug=False):
        self.data_path = data_path
        self.mode = mode

        self.bag_size = bag_size
        self.truncated_length = truncated_length
        self.coder_truncated_length = coder_truncated_length

        self.bert_tok = AutoTokenizer.from_pretrained(bert_path)
        if coder_path:
            self.coder_tok = AutoTokenizer.from_pretrained(coder_path)
        else:
            self.coder_tok = self.bert_tok

        self.path = os.path.join(data_path, mode)

        self.rel2id_path = os.path.join(data_path, 'rel2id.json')
        with open(self.rel2id_path, 'r') as f:
            self.rel2id = json.load(f)

        self.load(self.path, debug)

    def load(self, path, debug=False):
        f = open(path)

        self.bag_scope = []
        # self.name2id = {}
        self.bag_name = []
        self.bag_count = {}
        self.facts = {}

        idx = -1
        for line in tqdm(f):
            if debug:
                if idx >= 100000:
                    break
            idx += 1
            line = line.rstrip()
            if len(line) > 0:
                df = eval(line)
                cuis = "|".join([df['h']['id'], df['t']['id']])
            # if cuis not in self.name2id:
            #     self.name2id[cuis] = len(self.name2id)
            #     self.bag_scope.append([])
            #     self.bag_name.append(cuis)
            # if len(self.bag_scope[self.name2id[cuis]]) < 10 * self.bag_size:
            #     self.bag_scope[self.name2id[cuis]].append(df)
            if self.bag_count.get(cuis, 0) < self.bag_size:
                self.bag_scope.append([df])
                self.bag_name.append(cuis)
                if not cuis in self.bag_count:
                    self.bag_count[cuis] = 0
                self.bag_count[cuis] += 1
        f.close()

        self.len = len(self.bag_scope)
        
        return

if __name__ == '__main__':
    dev_dataset = REDataset('data/sample_data', 'dev', 3, 128, 30, 'bert-base-cased', 'GanjinZero/UMLSBert_ENG')
    xxx = dev_dataset[0]
    from torch.utils.data import DataLoader
    dev_dataloader = DataLoader(dev_dataset, batch_size=2, collate_fn=my_collate_fn, shuffle=False)
    for batch in dev_dataloader:
        print(batch)
        import ipdb; ipdb.set_trace()
