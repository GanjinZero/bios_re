# coding:utf-8
import sys, json
import torch
import os
import numpy as np
#import opennre
sys.path.append('/raid/zheng')
from OpenNRE import opennre
import argparse
import logging
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-uncased', 
        help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='', 
        help='Checkpoint name')
parser.add_argument('--result', default='', 
        help='Result name')
parser.add_argument('--pooler', default='entity', choices=['cls', 'entity'], 
        help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true', 
        help='Only run test')
parser.add_argument('--mask_entity', action='store_true', 
        help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc'],
        help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['none', 'wiki_distant', 'nyt10', 'nyt10m', 'wiki20m'],
        help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
        help='Training data file')
parser.add_argument('--val_file', default='', type=str,
        help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
        help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
        help='Relation to ID file')

# Bag related
parser.add_argument('--bag_size', type=int, default=4,
        help='Fixed bag size. If set to 0, use original bag sizes')

# Hyper-parameters
parser.add_argument('--batch_size', default=16, type=int,
        help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
        help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
        help='Maximum sentence length')
parser.add_argument('--max_epoch', default=3, type=int,
        help='Max number of training epochs')

# Exp
parser.add_argument('--aggr', default='att', choices=['one', 'att', 'avg'])


# Seed
parser.add_argument('--seed', default=42, type=int,
        help='Seed')

args = parser.parse_args()

# Set random seed
set_seed(args.seed)
args = parser.parse_args()

# Some basic settings
root_path = '.'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

device = "cuda:0"

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))


# Define the sentence encoder
if args.pooler == 'entity':
    sentence_encoder = opennre.encoder.BERTEntityEncoder(
        max_length=args.max_length, 
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
elif args.pooler == 'cls':
    sentence_encoder = opennre.encoder.BERTEncoder(
        max_length=args.max_length, 
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
else:
    raise NotImplementedError


# Define the model
if args.aggr == 'att':
    model = opennre.model.BagAttention(sentence_encoder, len(rel2id), rel2id)
elif args.aggr == 'avg':
    model = opennre.model.BagAverage(sentence_encoder, len(rel2id), rel2id)
elif args.aggr == 'one':
    model = opennre.model.BagOne(sentence_encoder, len(rel2id), rel2id)
else:
    raise NotImplementedError

# Define the whole training framework
framework = opennre.framework.BagRE(
    train_path=args.train_file,
    val_path=args.val_file,
    test_path=args.test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt="adamw",
    bag_size=args.bag_size
)

# Test the model
framework.load_state_dict(torch.load(ckpt)['state_dict'])
framework.eval()
model = framework.model.module.to(device)
def infer(model, bag):
    model.eval()
    tokens = []
    pos1s = []
    pos2s = []
    masks = []
    for item in bag:
        token, pos1, pos2, mask = model.sentence_encoder.tokenize(item)
        tokens.append(token)
        pos1s.append(pos1)
        pos2s.append(pos2)
        masks.append(mask)
    tokens = torch.cat(tokens, 0).unsqueeze(0).to(device) # (n, L)
    pos1s = torch.cat(pos1s, 0).unsqueeze(0).to(device)
    pos2s = torch.cat(pos2s, 0).unsqueeze(0).to(device)
    masks = torch.cat(masks, 0).unsqueeze(0).to(device)
    scope = torch.tensor([[0, len(bag)]]).long().to(device) # (1, 2)
    bag_logits = model.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0) # (N) after softmax
    """
    score, pred = bag_logits.max(0)
    score = score.item()
    pred = pred.item()
    rel = model.id2rel[pred]
    """
    score, pred = bag_logits.topk(k=3)
    score = [s.item() for s in score]
    pred = [s.item() for s in pred]
    rel = [model.id2rel[p] for p in pred]
    return (rel, score)

"""
bag = [{'text': 'In insulin-dependent diabetes mellitus (IDDM) the situation is less clear, but a decrement of the circadian glucose profile has been shown.', 'relation': 'CHD\tisa', 'h': {'pos': [21, 38], 'id': 'C0011849', 'name': 'diabetes mellitus'}, 't': {'pos': [3, 20], 'id': 'C0011854', 'name': 'insulin-dependent'}}]

bag2 = [bag[0], bag[0]]

print(infer(model, bag))
print(infer(model, bag2))
"""
with open('../../human_eval/sort_test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
lines = [eval(line) for line in lines]

from tqdm import tqdm

last_htrel = None

label_tag_l = []
text_l = []
h_l = []
h_l_id = []
h_l_pos = []
t_l = []
t_l_id = []
t_l_pos = []
label_relation = []
rec0 = []
s0 = []
rec1 = []
s1 = []
rec2 = []
s2 = []
high_l = []
newrel_l = ["" for _ in range(100)]
reason_l = ["" for _ in range(100)]


def m(x):
    return "|".join([str(a) for a in x])

d = dict()
h_rel = dict()
h_score = dict()
#for idx in range(100):
from tqdm import trange
for idx in trange(len(lines)):
    line = lines[idx]
    rel, score = infer(model, [lines[idx]])
    htrel = line['h']['id'] + line['t']['id'] + line['relation']
    d[idx] = (rel, score)
    if htrel not in h_rel:
        h_rel[htrel] = []
        h_score[htrel] = []
    h_rel[htrel].extend(rel)
    h_score[htrel].extend(score)

#for idx in range(100):
for idx in trange(len(lines)):
    line = lines[idx]
    htrel = line['h']['id'] + line['t']['id'] + line['relation']
    rel, score = d[idx]
    label_tag = htrel != last_htrel
    last_htrel = htrel
    high = 0

    rec0.append(rel[0])
    s0.append(score[0])
    rec1.append(rel[1])
    s1.append(score[1])
    rec2.append(rel[2])
    s2.append(score[2])
    
    label_tag_l.append(label_tag)
    text_l.append(line['text'])
    h_l.append(line['h']['name'])
    h_l_id.append(line['h']['id'])
    h_l_pos.append(m(line['h']['pos']))
    t_l.append(line['t']['name'])
    t_l_id.append(line['t']['id'])
    t_l_pos.append(m(line['t']['pos']))
    label_relation.append(line['relation'])
    high_l.append(high)

import pandas as pd

label_high = []
label_low = []
text_high = []
text_low = []
head_high = []
head_low = []
head_id_high = []
heae_id_low = []
tail_high = []
tail_low = []
tail_id_high = []
tail_id_low = []
rec0_high = []
s0_high = []
rec1_high = []
s1_high = []
rec1_high = []
s1_high = []
rec0_low = []
s0_low = []
rec1_low = []
s1_low = []
rec1_low = []
s1_low = []

all_csv = pd.DataFrame({'new_rel':["" for _ in range(len(label_tag_l))],
                        'sentence_mark':["" for _ in range(len(label_tag_l))],
                        'BAG_START':label_tag_l,
                        'text':text_l,
                        'head':h_l,
                        'tail':t_l,
                        'dist_rel':label_relation,
                        'rec0':rec0,
                        's0':s0,
                        'rec1':rec1,
                        's1':s1,
                        'rec2':rec2,
                        's2':s2,
                        'hc':h_l_id,
                        'tc':t_l_id,
                        'hp':h_l_pos,
                        'tp':t_l_pos})

all_csv.to_csv('all_full.csv')
print(sum(label_tag_l))
"""
high_csv = pd.DataFrame({'new_relation':["" for _ in len(label_tab_l)]})
low_csv = 
high_csv.to_csv('high.csv')
low_csv.to_csv('low.csv')
"""
