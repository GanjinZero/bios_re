import torch
from torch import nn
from transformers import AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup


class RelationExtractionEncoder(nn.Module):
    def __init__(self, bert_name, pooler='entity',
                       coder_name=None, coder_pooler='cls', coder_freeze=True,
                       rep_dropout=0.1):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_name)
        self.hidden_dim = AutoConfig.from_pretrained(bert_name).hidden_size

        self.pooler = pooler
        if self.pooler == 'cls':
            self.input_dim = self.hidden_dim
        elif self.pooler == 'entity':
            self.input_dim = 2 * self.hidden_dim
        
        if coder_name:
            self.coder_dim = AutoConfig.from_pretrained(coder_name).hidden_size
            self.coder = AutoModel.from_pretrained(coder_name)
            self.coder_pooler = coder_pooler
            if coder_freeze:
                for param in self.coder.parameters():
                    param.requires_grad = False

            self.input_dim += 2 * self.coder_dim

        self.linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout = nn.Dropout(rep_dropout)

    def forward(self, input_ids, attention_mask, ent0_pos=None, ent1_pos=None,
                      ent0_input_ids=None, ent0_att=None, ent1_input_ids=None, ent1_att=None):
        hidden = self.bert(input_ids, attention_mask)
        
        bsz = ent0_input_ids.shape[0]
        if self.pooler == 'cls':
            sen_rep = hidden[1] # batch * hidden
        elif self.pooler == 'entity':
            idx = torch.arange(0,bsz).long().to(input_ids.device)
            ent0_rep = hidden[0][idx,ent0_pos]
            ent1_rep = hidden[0][idx,ent1_pos]
            sen_rep = torch.cat([ent0_rep, ent1_rep], dim=1)

        if hasattr(self, 'coder'):
            
            if self.coder_pooler == 'cls':
                coder_hidden = self.coder(torch.cat([ent0_input_ids, ent1_input_ids], dim=0), 
                                      attention_mask=torch.cat([ent0_att, ent1_att], dim=0))[1] # 2B, H
            else:
                raise NotImplementedError
            head_coder = coder_hidden[0:bsz]
            tail_coder = coder_hidden[bsz:]
            sen_rep = torch.cat([sen_rep, head_coder, tail_coder], dim=1) # B * ?H

        final = self.linear(sen_rep)
        final = self.dropout(final)

        return final

class BagRE(nn.Module):
    def __init__(self, encoder, aggr, class_count, criterion):
        super().__init__()
        self.encoder = encoder
        self.aggr = aggr
        if self.aggr == 'att':
            self.att_query = nn.Linear(self.encoder.hidden_dim, self.class_count)
        self.class_count = class_count
        self.classifier = nn.Linear(self.encoder.hidden_dim, self.class_count)

        self.criterion = criterion
        if self.criterion == 'softmax':
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.criterion == 'binary':
            self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward_logits(self, bag_id, 
                      input_ids, attention_mask, ent0_pos=None, ent1_pos=None,
                      ent0_input_ids=None, ent0_att=None, ent1_input_ids=None, ent1_att=None):

        hidden = self.encoder(input_ids, attention_mask, ent0_pos, ent1_pos,
                              ent0_input_ids, ent0_att, ent1_input_ids, ent1_att) # batch * hidden

        bag_id_set = [] # use list to preserve order
        bag_first_id = []
        for idx, id in enumerate(bag_id):
            if id.item() not in bag_id_set:
                bag_id_set.append(id.item())
                bag_first_id.append(idx)
        bag_first_id = torch.LongTensor(bag_first_id).to(bag_id.device)
                    
        bsz = input_ids.shape[0]
        if self.aggr == 'one':
            logits = self.classifier(hidden) # batch * class
            
            if self.criterion == 'softmax':
                raise NotImplementedError
            elif self.criterion == 'binary':
                select_logits = []
                for id in bag_id_set:
                    id_logits = logits[bag_id==id] # now_bag * class
                    select_logits.append(id_logits.max(dim=0)[0])
                select_logits = torch.stack(select_logits, dim=0)

        elif self.aggr == 'mean':
            select_hidden = []
            for id in bag_id_set:
                id_hidden = hidden[bag_id==id] # now_bag * hidden
                id_hidden_mean = torch.mean(id_hidden, dim=0) # hidden
                select_hidden.append(id_hidden_mean)
            select_hidden = torch.stack(select_hidden, dim=0)
            select_logits = self.classifier(select_hidden)

        elif self.aggr == 'att':
            # TODO: Finish This Code
            raise NotImplementedError
            score = self.att_query(hidden) # batch * class
            select_hidden = []
            for id in bag_id_set:
                id_hidden = hidden[bag_id==id] # now_bag * hidden
                id_score = score[bag_id==id] # now_bag * class
                id_class_hidden = torch.mm(id_score.t(), id_hidden) # class * hidden
                select_hidden.append(id_class_hidden)
            select_hidden = torch.stack(select_hidden, dim=0) # bag * class * hidden
            # select_logits = torch.matmul(select_hidden, self.classifier.weight.t())

        return select_logits, bag_first_id # bag * class

    def forward(self, bag_id, 
                      input_ids, attention_mask, ent0_pos=None, ent1_pos=None,
                      ent0_input_ids=None, ent0_att=None, ent1_input_ids=None, ent1_att=None, labels=None):
        logits, bag_first_id = self.forward_logits(bag_id, 
                      input_ids, attention_mask, ent0_pos, ent1_pos,
                      ent0_input_ids, ent0_att, ent1_input_ids, ent1_att)
        if self.criterion == 'softmax':
            return self.loss_fn(logits, labels[bag_first_id])
        else:
            ones = torch.eye(self.class_count).to(bag_id.device)
            label = ones.index_select(0, labels[bag_first_id])
            return self.loss_fn(logits, label)

    def configure_optimizers(self, args, train_dataloader):
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay,
             'lr': args.learning_rate},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': args.learning_rate}
        ]
        optimizer = AdamW(params, eps=args.adam_epsilon)

        total_steps = len(train_dataloader) * args.train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=int(total_steps * args.warmup_ratio),
                                                    num_training_steps=total_steps)
        return [optimizer], [scheduler]
    