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
                     output_path='../../data/1117_v2/'):

    try:
        os.system(f'mkdir {output_path}')
    except BaseException:
        pass

    train_set = set()
    dev_set = set()
    test_set = set()
    u_count = 0

    for line in tqdm(open(rel_path, 'r')):
        rel_line = eval(line.strip())
        cui0 = rel_line['h']['id']
        cui1 = rel_line['t']['id']
        cuis = "|".join([cui0, cui1])
        if not (cuis in train_set or cuis in dev_set or cuis in test_set):
            u_count += 1
            if u_count % 100 == 47:
                dev_set.update([cuis])
            elif u_count % 100 == 72:
                test_set.update([cuis])
            else:
                train_set.update([cuis])
        if cuis in train_set:
            with open(os.path.join(output_path, 'train.txt'), 'a+') as f:
                f.write(line.strip() + "\n")
        elif cuis in dev_set:
            with open(os.path.join(output_path, 'dev.txt'), 'a+') as f:
                f.write(line.strip() + "\n")
        elif cuis in test_set:
            with open(os.path.join(output_path, 'test.txt'), 'a+') as f:
                f.write(line.strip() + "\n")
   
if __name__ == '__main__':
    train_test_split()
