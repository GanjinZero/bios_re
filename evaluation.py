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

with open(os.path.join(data_path, 'rel2id.json'), 'r') as f:
    rel2id = json.load(f)
id2rel = {id:rel for rel, id in rel2id.items()}


def evaluate(model_path):
    with open(os.path.join(os.path.dirname(model_path), 'args.json'), 'r') as f:
        config = json.load(f)
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

    yhat_raw = []
    y = []

    for batch_idx, batch in tqdm(enumerate(epoch_iter)):
        batch_gpu = tuple([x.to(device) for x in batch])
        with torch.no_grad():
            select_logits, bag_first_id = model.forward_logits(bag_id=batch_gpu[0], 
                        input_ids=batch_gpu[1], attention_mask=batch_gpu[2], ent0_pos=batch_gpu[3], ent1_pos=batch_gpu[4],
                        ent0_input_ids=batch_gpu[5], ent0_att=batch_gpu[6], ent1_input_ids=batch_gpu[7], ent1_att=batch_gpu[8]) # bat_count * class_count
            select_label = batch_gpu[9][bag_first_id] # bag_count
        yhat_raw.append(select_logits.cpu().detach())
        y.append(select_label.cpu().detach())

    yhat_raw = torch.cat(yhat_raw, dim=0)
    y = torch.cat(y, dim=0)

    yhat = (yhat_raw >= 0).float()
    metric, class_metric = calculate_metric(yhat_raw, y, yhat)
    print(metric)
    print(class_metric)

    epoch_nb = os.path.basename(model_path).split('.')[0]
    with open(os.path.join(os.path.dirname(model_path), f'{epoch_nb}_metric.json'), 'w') as f:
        json.dump(metric, f, indent=2)
    with open(os.path.join(os.path.dirname(model_path), f'{epoch_nb}_class_metric.json'), 'w') as f:
        json.dump(class_metric, f, indent=2)

    return


def calculate_metric(yhat_raw, y, yhat):
    metric = {}
    class_metric = {}
    class_count = yhat_raw.shape[1]

    # micro-f1
    #label_count = y.shape[0]
    label_count = y.sum()
    predict_count = yhat.sum()
    #ones = torch.eye(class_count)
    #binary_label = ones.index_select(0, y)
    binary_label = y
    true_count = (yhat * binary_label).sum()
    precision = true_count / predict_count
    recall = true_count / label_count
    if true_count == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall / (precision + recall)).item()
    metric['micro-p'] = precision.item()
    metric['micro-r'] = recall.item()
    metric['micro-f1'] = f1

    # macro-f1
    macro_f1_list = []
    for i in range(class_count):
        #yhat_i = yhat[y==i]
        #binary_label_i = binary_label[y==i]
        yhat_i = yhat[:,i]
        binary_label_i = binary_label[:,i]
        label_i_count = binary_label_i.sum()
        predict_i_count = yhat_i.sum()
        true_i_count = (yhat_i * binary_label_i).sum()
        precision_i = true_i_count / predict_i_count
        recall_i = true_i_count / label_i_count
        if true_i_count == 0:
            f1_i = 0
        else:
            f1_i = (2 * precision_i * recall_i / (precision_i + recall_i)).item()
        class_metric[f'p_{id2rel[i]}'] = precision_i.item()
        class_metric[f'r_{id2rel[i]}'] = recall_i.item()
        class_metric[f'f1_{id2rel[i]}'] = f1_i
        macro_f1_list.append(f1_i)
    metric['macro-f1'] = sum(macro_f1_list) / len(macro_f1_list)
    return metric, class_metric


#evaluate('/media/sda1/GanjinZero/bios_re/output/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_coder_debug/epoch1.pth')
#evaluate('output_1203/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_coder/epoch2.pth')
evaluate('output_1203_na10/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_coder/epoch2.pth')
