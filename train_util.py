import os
import torch
import numpy as np
import json
from model.model import RelationExtractionEncoder, BagRE

def get_output_folder_name(args):
    config = [args.pooler, args.coder_pooler, args.coder_freeze, args.rep_dropout, args.aggr, args.criterion, args.train_epochs,
              args.learning_rate, args.bag_size, args.batch_size * args.n_gpu * args.gradient_accumulation_steps, args.warmup_ratio]
    if args.reverse_train:
        config.append('rev')
    if args.limit_dis is not None:
        config.append(str(args.limit_dis))
    if args.coder_path:
        config.append('coder')
    if args.debug:
        config.append('debug')
    return "_".join([str(x) for x in config])

def get_model(args):
    with open(os.path.join(args.data_path, 'rel2id.json'), 'r') as f:
        rel2id = json.load(f)
    class_count = len(rel2id)
    encoder = RelationExtractionEncoder(args.bert_path, args.pooler, args.coder_path, args.coder_pooler, args.coder_freeze, args.rep_dropout)
    model = BagRE(encoder, args.aggr, class_count, args.criterion)
    return model
