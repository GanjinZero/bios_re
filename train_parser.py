import argparse


def get_train_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_gpu', type=int, default=1)

    parser.add_argument('--output_base_dir', type=str, default='./output')
    parser.add_argument('--data_path', type=str, default='./data/1117_v2') #./data/1117_v2   sample_data

    parser.add_argument('--truncated_length', type=str, default=128)
    parser.add_argument('--coder_truncated_length', type=str, default=32)
    
    parser.add_argument('--bert_path', type=str, default='../plm/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--pooler', type=str, default='entity', choices=['entity', 'cls'])
    parser.add_argument('--coder_path', type=str, default='../plm/coder_eng')
    parser.add_argument('--coder_pooler', type=str, default='cls', choices=['cls', 'mean'])
    parser.add_argument('--coder_freeze', action='store_true')
    
    parser.add_argument('--rep_dropout', type=float, default=0.1)
    
    parser.add_argument('--aggr', type=str, default='one')
    parser.add_argument('--criterion', type=str, default='binary')
    
    parser.add_argument('--reverse_train', action='store_true')
    parser.add_argument('--limit_dis', type=str, default=None)
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--bag_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)

    parser.add_argument('--tag', type=str, default='')

    return parser
    
