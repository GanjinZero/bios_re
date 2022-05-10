import os
from tqdm import tqdm

# manual test 
# train all sentences
# test 20w sentences
# valid 20w sentences

def check(line):
    return line.startswith('{') and line.strip().endswith('}')

def get_cui(json_line):
    d = eval(json_line.strip())
    cui0 = d['h']['id']
    cui1 = d['t']['id']
    return "|".join([cui0, cui1])

def get_rel(json_line):
    d = eval(json_line.strip())
    return d['relation']

def output(rel_lines, na_lines, 
           filename_list, output_path,
           ignore_cui_pair=None):
    print(output_path)
    cui_pair = {}
    with open(output_path, "w", encoding="utf-8") as f:
        for filename in tqdm(filename_list):
            rel = rel_lines[filename]
            na = na_lines[filename]

            for r in rel:
                strcui = get_cui(r)
                if ignore_cui_pair is not None and strcui in ignore_cui_pair:
                    continue
                if not strcui in cui_pair:
                    cui_pair[strcui] = []
                cui_pair[strcui].append(r)

            for r in na:
                strcui = get_cui(r)
                if ignore_cui_pair is not None and strcui in ignore_cui_pair:
                    continue
                if not strcui in cui_pair:
                    cui_pair[strcui] = []
                cui_pair[strcui].append(r)

        line_count = 0
        nona_cuip_count = 0
        nona_line_count = 0
        cui_set = set()
        rel_set = set()
        for strcui in tqdm(cui_pair):
            cui0, cui1 = strcui.split('|')
            cui_set.update([cui0, cui1])
            for line in cui_pair[strcui]:
                f.write(line + "\n")
                line_count += 1
                rel = get_rel(line)
                rel_set.update([rel])
                if rel != "NA":
                    nona_line_count += 1
            if rel != "NA":
                nona_cuip_count += 1

        print(f'CUI count: {len(cui_set)}')
        print(f'Rel count: {len(rel_set)}')
        print(f'CUI-pair count: {len(cui_pair)}')
        print(f'None NA CUI-pair count:{nona_cuip_count}')
        print(f'Lines count: {line_count}')
        print(f'None NA Lines count:{nona_line_count}')
        return cui_pair.keys()

def train_test_split(rel_path='../../data/sentence_coder_1117.json',
                     na_path='',
                     output_path='../../data/1117_v2/'):

    try:
        os.system(f'mkdir {output_path}')
    except BaseException:
        pass

    train_set = set()
    dev_set = set()
    test_set = set()
    u_count = 0

    train_idx = []
    dev_idx = []
    test_idx = []
    for line in tqdm(open(rel_path, 'r')):
        rel_line = eval(line.strip())
        cui0 = rel_line['h']['id']
        cui1 = rel_line['t']['id']
        cuis = "|".join([cui0, cui1])
        rel = rel_line['relation']
        if cuis in train_set or rel == ['NA']:
            train_idx.append(line)
        elif cuis in dev_set:
            dev_idx.append(line)
        elif cuis in test_set:
            test_idx.append(line)
        else:
            u_count += 1
            if u_count % 100 == 47:
                dev_set.update([cuis])
                dev_idx.append(line)
            elif u_count % 100 == 72:
                test_set.update([cuis])
                test_idx.append(line)
            else:
                train_set.update([cuis])
                train_idx.append(line)
    """            
    with open(os.path.join(output_path, 'train.txt'), 'w') as f:
        for line in train_idx:
            f.write(line.strip() + "\n")
    with open(os.path.join(output_path, 'test.txt'), 'w') as f:
        for line in test_idx:
            f.write(line.strip() + "\n")
    with open(os.path.join(output_path, 'dev.txt'), 'w') as f:
        for line in dev_idx:
            f.write(line.strip() + "\n")
    """

    if na_path:
        train_idx = []
        test_idx = []
        dev_idx = []
        for line in tqdm(open(na_path, 'r')):
            rel_line = eval(line.strip())
            cui0 = rel_line['h']['id']
            cui1 = rel_line['t']['id']
            if cuis in train_set:
                train_idx.append(line)
            elif cuis in dev_set:
                dev_idx.append(line)
            elif cuis in test_set:
                test_idx.append(line)
            else:
                u_count += 1
                if u_count % 100 == 47:
                    dev_set.update([cuis])
                    dev_idx.append(line)
                elif u_count % 100 == 72:
                    test_set.update([cuis])
                    test_idx.append(line)
                else:
                    train_set.update([cuis])
                    train_idx.append(line)

    with open(os.path.join(output_path, 'train.txt'), 'a+') as f:
        for line in train_idx:
            f.write(line.strip() + "\n")
    with open(os.path.join(output_path, 'test.txt'), 'a+') as f:
        for line in test_idx:
            f.write(line.strip() + "\n")
    with open(os.path.join(output_path, 'dev.txt'), 'a+') as f:
        for line in dev_idx:
            f.write(line.strip() + "\n")


if __name__ == '__main__':
    #train_test_split('../data/1202/raw.json', '../data/1203/raw_na.json', '../data/1203')
    #train_test_split('../data/1202/raw.json', '../data/1203/raw_na_10.json', '../data/1203_na10')
    train_test_split('../data/1221/dataset.json', '', '../data/1221')
