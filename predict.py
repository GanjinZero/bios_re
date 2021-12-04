import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import torch
from torch import nn 
from tqdm import tqdm
import shutil, json, ipdb, sys
import numpy as np
from torch.utils.data import DataLoader
from data_util import REDataset, my_collate_fn


device = 'cuda'
data_path = './data/1203'

MAX_COUNT = 10

with open(os.path.join(data_path, 'rel2id.json'), 'r') as f:
    rel2id = json.load(f)
id2rel = {id:rel for rel, id in rel2id.items()}


def predict_one(model_path):
    with open(os.path.join(os.path.dirname(model_path), 'args.json'), 'r') as f:
        config = json.load(f)
    assert config['aggr'] == 'one'
    bag_size = config['bag_size']
    batch_size = 16
    truncated_length = config['truncated_length']
    coder_truncated_length = config['coder_truncated_length']
    bert_path = config['bert_path']
    coder_path = config['coder_path']

    test_datast = REDataset(data_path, 'test', bag_size, truncated_length, coder_truncated_length, bert_path, coder_path)
    test_dataloader = DataLoader(test_datast, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1, pin_memory=True)

    model = torch.load(model_path).to(device)
    model.eval()

    epoch_iter = tqdm(test_dataloader)

    example_dict = {rel:[] for rel, id in rel2id.items()}
    cnt = 0

    for batch_idx, batch in tqdm(enumerate(epoch_iter)):
        if cnt >= 20 * MAX_COUNT - 1:
            break
        batch_gpu = tuple([x.to(device) for x in batch])
        with torch.no_grad():
            hidden = model.encoder(batch_gpu[1], batch_gpu[2], batch_gpu[3], batch_gpu[4],
                              batch_gpu[5], batch_gpu[6], batch_gpu[7], batch_gpu[8]) # batch * hidden
            logits = model.classifier(hidden) # batch * class
            label = batch_gpu[9]

            bsz = logits.shape[0]
            for i in range(bsz):
                for j in range(len(id2rel)):
                    if logits[i][j] > 0.0:
                        rel = id2rel[j]
                        if len(example_dict[rel]) > MAX_COUNT:
                            continue
                        text = test_datast.bert_tok.convert_ids_to_tokens(batch_gpu[1][i][batch_gpu[2][i]==1])
                        text = test_datast.bert_tok.convert_tokens_to_string(text)
                        ent0 = test_datast.coder_tok.convert_ids_to_tokens(batch_gpu[5][i][batch_gpu[6][i]==1][2:-2])
                        ent0 = test_datast.bert_tok.convert_tokens_to_string(ent0)
                        ent1 = test_datast.coder_tok.convert_ids_to_tokens(batch_gpu[7][i][batch_gpu[8][i]==1][2:-2])
                        ent1 = test_datast.bert_tok.convert_tokens_to_string(ent1)
                        #true_rel = [id2rel[x.item()] for x in label[i]]
                        true_rel = []
                        for idx, k in enumerate(label[i]):
                            if k.item() > 0:
                                true_rel.append(id2rel[idx])
                        pos1 = batch_gpu[3][i]
                        pos2 = batch_gpu[4][i]
                        if pos1 < pos2:
                            rev = False
                        elif pos1 > pos2:
                            rev = True
                        else:
                            continue
                        if not rev:
                            example_dict[rel].append({'text':text, 'h':ent0, 't':ent1, 'label':true_rel})
                        else:
                            example_dict[rel].append({'text':text, 'h':ent1, 't':ent0, 'label':true_rel})
                        cnt += 1


    epoch_nb = os.path.basename(model_path).split('.')[0]
    with open(os.path.join(os.path.dirname(model_path), f'{epoch_nb}_example_dict.json'), 'w') as f:
        json.dump(example_dict, f, indent=2)

    return

#predict_one('/media/sda1/GanjinZero/bios_re/output/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0/epoch2.pth')
predict_one('output_1203/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_coder/epoch2.pth')
