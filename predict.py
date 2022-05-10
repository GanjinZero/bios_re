import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import torch
from torch import nn 
from tqdm import tqdm
import shutil, json, ipdb, sys
import numpy as np
from torch.utils.data import DataLoader
from data_util import REDataset, PredictOneDataset, my_collate_fn


device = 'cuda'
data_path = './data/1224'

MAX_COUNT = 20

with open(os.path.join(data_path, 'rel2id.json'), 'r') as f:
    rel2id = json.load(f)
if os.path.exists(os.path.join(data_path, 'id2rel.json')):
    with open(os.path.join(data_path, 'id2rel.json'), 'r') as f:
        id2rel = json.load(f)
    id2rel = {int(id):rel for id, rel in id2rel.items()}
else:
    id2rel = {id:rel for rel, id in rel2id.items()}


def predict_one(model_path, test_dataloader, output_path, demo=False):
    with open(os.path.join(os.path.dirname(model_path), 'args.json'), 'r') as f:
        config = json.load(f)
    assert config['aggr'] == 'one'
    bag_size = config['bag_size']
    batch_size = 16
    truncated_length = config['truncated_length']
    coder_truncated_length = config['coder_truncated_length']
    bert_path = config['bert_path']
    coder_path = config['coder_path']

    model = torch.load(model_path).to(device)
    model.eval()

    epoch_iter = tqdm(test_dataloader)

    if demo:
        example_dict = {rel:[] for _, rel in id2rel.items()}
    else:
        opt = {}
    cnt = 0

    for batch_idx, batch in tqdm(enumerate(epoch_iter)):
        if demo:
            if cnt >= 20 * MAX_COUNT - 1:
                break
        batch_gpu = tuple([x.to(device) for x in batch])
        with torch.no_grad():
            hidden = model.encoder(batch_gpu[1], batch_gpu[2], batch_gpu[3], batch_gpu[4],
                              batch_gpu[5], batch_gpu[6], batch_gpu[7], batch_gpu[8]) # batch * hidden
            logits = model.classifier(hidden) # batch * class
            label = batch_gpu[9]

            j_idx = torch.arange(logits.shape[1]).to(device)

            bsz = logits.shape[0]
            for i in range(bsz):
                if demo:
                    for j in range(len(id2rel)):
                        if logits[i][j] > 0.0:
                            rel = id2rel[j]
                            if len(example_dict[rel]) > MAX_COUNT:
                                continue
                            text = test_dataloader.dataset.bert_tok.convert_ids_to_tokens(batch_gpu[1][i][batch_gpu[2][i]==1])
                            text = test_dataloader.dataset.bert_tok.convert_tokens_to_string(text)
                            ent0 = test_dataloader.dataset.coder_tok.convert_ids_to_tokens(batch_gpu[5][i][batch_gpu[6][i]==1][2:-2])
                            ent0 = test_dataloader.dataset.bert_tok.convert_tokens_to_string(ent0)
                            ent1 = test_dataloader.dataset.coder_tok.convert_ids_to_tokens(batch_gpu[7][i][batch_gpu[8][i]==1][2:-2])
                            ent1 = test_dataloader.dataset.bert_tok.convert_tokens_to_string(ent1)
                            #true_rel = [id2rel[x.item()] for x in label[i]]
                            true_rel = []
                            for idx, k in enumerate(label[i]):
                                if k.item() > 0:
                                    true_rel.append(id2rel[idx])
                            #use follow to only see false examples
                            #if rel in true_rel:
                            #    continue
                            pos1 = batch_gpu[3][i]
                            pos2 = batch_gpu[4][i]
                            if pos1 < pos2:
                                rev = False
                            elif pos1 > pos2:
                                rev = True
                            if not rev:
                                example_dict[rel].append({'text':text, 'h':ent0, 't':ent1, 'label':true_rel})
                            else:
                                example_dict[rel].append({'text':text, 'h':ent1, 't':ent0, 'label':true_rel})
                            cnt += 1
                else:
                    choose_j = j_idx[logits[i] > 0]
                    if choose_j.shape[0] > 0:
                        text = test_dataloader.dataset.bert_tok.convert_ids_to_tokens(batch_gpu[1][i][batch_gpu[2][i]==1])
                        text = test_dataloader.dataset.bert_tok.convert_tokens_to_string(text)
                        ent0 = test_dataloader.dataset.coder_tok.convert_ids_to_tokens(batch_gpu[5][i][batch_gpu[6][i]==1][2:-2])
                        ent0 = test_dataloader.dataset.bert_tok.convert_tokens_to_string(ent0)
                        ent1 = test_dataloader.dataset.coder_tok.convert_ids_to_tokens(batch_gpu[7][i][batch_gpu[8][i]==1][2:-2])
                        ent1 = test_dataloader.dataset.bert_tok.convert_tokens_to_string(ent1)
                        rel = [id2rel[j.item()] for j in choose_j]
                        pos1 = batch_gpu[3][i]
                        pos2 = batch_gpu[4][i]
                        if pos1 < pos2:
                            rev = False
                        elif pos1 > pos2:
                            rev = True
                        with open(output_path, 'a+') as f:
                            if not rev:
                                strs = json.dumps({'text':text, 'h':ent0, 't':ent1, 'predict_rel':rel})
                            else:
                                strs = json.dumps({'text':text, 'h':ent1, 't':ent0, 'predict_rel':rel})
                            #strs = json.dumps({'text':text, 'h':ent1, 't':ent0, 'predict_rel':rel})
                            f.write(strs.strip() + "\n")

    # epoch_nb = os.path.basename(model_path).split('.')[0]
    if demo:
        with open(output_path, 'w') as f:
            json.dump(example_dict, f, indent=2)

    return


def eval_test(model_path):
    with open(os.path.join(os.path.dirname(model_path), 'args.json'), 'r') as f:
        config = json.load(f)
    assert config['aggr'] == 'one'
    bag_size = config['bag_size']
    batch_size = 16
    truncated_length = config['truncated_length']
    coder_truncated_length = config['coder_truncated_length']
    bert_path = config['bert_path']
    coder_path = config['coder_path']

    output_path = os.path.join(os.path.dirname(model_path), 'test_predict.json')
    test_datast = REDataset(data_path, 'test', bag_size, truncated_length, coder_truncated_length, bert_path, coder_path)
    test_dataloader = DataLoader(test_datast, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1, pin_memory=True)
    predict_one(model_path, test_dataloader, output_path, demo=True)

def predict_all(model_path, input_name=None, output_path=None):
    with open(os.path.join(os.path.dirname(model_path), 'args.json'), 'r') as f:
        config = json.load(f)
    assert config['aggr'] == 'one'
    bag_size = config['bag_size']
    batch_size = 64
    truncated_length = config['truncated_length']
    coder_truncated_length = config['coder_truncated_length']
    bert_path = config['bert_path']
    coder_path = config['coder_path']

    if output_path is None:
        output_path = os.path.join(os.path.dirname(model_path), 'all_predict.json')
        #output_path = os.path.join(os.path.dirname(model_path), 'example_opt.txt')

    input_name = 'inference_1125.json' if input_name is None else input_name
    test_datast = PredictOneDataset(data_path,
                                    input_name, bag_size, truncated_length, coder_truncated_length, bert_path, coder_path,
                                    debug=False)
    test_dataloader = DataLoader(test_datast, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1, pin_memory=True)
    predict_one(model_path, test_dataloader, output_path, demo=False)

if __name__ == "__main__":
    # eval_test('output_1203/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_coder/epoch2.pth')
    # predict_all('output_1203/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_coder/epoch2.pth')
    #eval_test('output_1203/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_rev_3,10_coder/epoch2.pth')
    #eval_test('output_1203_na10/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_rev_3,10_coder/epoch2.pth')
    #eval_test('output_1221/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_3,10_coder/epoch2.pth')
    #predict_all('output_1221/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_3,10_coder/epoch2.pth')
    eval_test('output_1224/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_3,10_coder/epoch2.pth')
    #import sys
    #input_name = sys.argv[1]
    #output_path = sys.argv[2]
    #predict_all('output_1224/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_3,10_coder/epoch2.pth', input_name, output_path)

