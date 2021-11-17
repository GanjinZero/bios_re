import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import torch
from torch import nn 
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import shutil, json, ipdb, sys
import numpy as np
from torch.utils.data import DataLoader
from train_parser import get_train_parser
from train_util import get_output_folder_name, get_model
from accelerate import DistributedDataParallelKwargs, Accelerator
from data_util import REDataset, my_collate_fn


def run(args):
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)

    output_basename = get_output_folder_name(args)
    accelerator.print(output_basename)
    output_path = os.path.join(args.output_base_dir, output_basename)

    try:
        os.system(f'mkdir -p {output_path}')
    except BaseException:
        pass

    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train_dataset = REDataset(args.data_path, 'train', args.bag_size, args.truncated_length, args.coder_truncated_length, args.bert_path, args.coder_path, args.debug)
    dev_dataset = REDataset(args.data_path, 'dev', args.bag_size, args.truncated_length, args.coder_truncated_length, args.bert_path, args.coder_path)
    test_datast = REDataset(args.data_path, 'test', args.bag_size, args.truncated_length, args.coder_truncated_length, args.bert_path, args.coder_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=True, num_workers=1, pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1, pin_memory=True)
    test_dataloader = DataLoader(test_datast, batch_size=args.batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1, pin_memory=True)

    model = get_model(args).to(accelerator.device)
    optimizer, scheduler_step = model.configure_optimizers(args, train_dataloader)
    optimizer = optimizer[0]
    scheduler_step = scheduler_step[0]

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    if accelerator.is_local_main_process and args.debug:
        pass

    steps = 0
    for epoch_idx in range(1, args.train_epochs + 1):
        epoch_dev_metric, epoch_test_metric, steps = train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler_step, args, accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            accelerator.save(accelerator.unwrap_model(model), os.path.join(output_path, f'epoch{epoch_idx}.pth'))
            # print metrics
            pass

def train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler_step, args, accelerator):
    model.train()
    epoch_loss = 0.

    epoch_iter = tqdm(train_dataloader, desc='Iteration', ascii=True, disable=not accelerator.is_local_main_process)
    for batch_idx, batch in enumerate(epoch_iter):
        batch_gpu = tuple([x.to(accelerator.device) for x in batch])
        # batch_gpu = {'bag_id':batch_gpu[0], 
        #               'input_ids':batch_gpu[1], 'attention_mask':batch_gpu[2], 'ent0_pos':batch_gpu[3], 'ent1_pos':batch_gpu[4],
        #               'ent0_input_ids':batch_gpu[5], 'ent0_att':batch_gpu[6], 'ent1_input_ids':batch_gpu[7], 'ent1_att':batch_gpu[8], 'labels':batch_gpu[9]}
        loss = model(bag_id=batch_gpu[0], 
                      input_ids=batch_gpu[1], attention_mask=batch_gpu[2], ent0_pos=batch_gpu[3], ent1_pos=batch_gpu[4],
                      ent0_input_ids=batch_gpu[5], ent0_att=batch_gpu[6], ent1_input_ids=batch_gpu[7], ent1_att=batch_gpu[8], labels=batch_gpu[9])

        batch_loss = float(loss.item())
        epoch_loss += batch_loss

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        accelerator.backward(loss)

        epoch_iter.set_description("Epoch: %0.4f, Batch: %0.4f" % (epoch_loss / (batch_idx + 1), batch_loss))

        if (steps + 1) % args.gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if scheduler_step is not None:
                scheduler_step.step()
            model.zero_grad()

        steps += 1

    accelerator.wait_for_everyone()

    return {}, {}, steps

def main():
    parser = get_train_parser()
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
