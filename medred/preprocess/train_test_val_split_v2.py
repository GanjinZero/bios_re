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

def train_test_split(rel_path='../rxnorm_snomedctus_medrt_rel',
                     na_path='../na_rel',
                     output_path='../dataset/'):

    try:
        os.system(f'mkdir {output_path}')
    except BaseException:
        pass

    rel_lines = {}
    rel_lines_count = {}
    na_lines = {}
    na_lines_count = {}

    print('Load Rel Data')
    filename_list = []
    for filename in tqdm(os.listdir(rel_path)):
        filename_list.append(filename)
        with open(os.path.join(rel_path, filename), "r", encoding="utf-8") as f:
            rel_lines[filename] = [line.strip() for line in f.readlines() if check(line)]
        rel_lines_count[filename] = len(rel_lines[filename])

    print('Load NA Data')
    for filename in tqdm(os.listdir(na_path)):
        with open(os.path.join(na_path, filename), "r", encoding="utf-8") as f:
            na_lines[filename] = [line.strip() for line in f.readlines() if check(line)]
        na_lines_count[filename] = len(na_lines[filename])
    
    test_filename_list = []
    test_sentence_count = 0
    dev_filename_list = []
    dev_sentence_count = 0
    for filename in filename_list[::-1]:
        if test_sentence_count < 100000:
            test_sentence_count += rel_lines_count[filename] + na_lines_count[filename]
            test_filename_list.append(filename)
        if test_sentence_count >= 100000 and dev_sentence_count < 500000:
            dev_sentence_count += rel_lines_count[filename] + na_lines_count[filename]
            dev_filename_list.append(filename)

    print('Dev')
    print(dev_filename_list)
    print('Test')
    print(test_filename_list)

    test_cui_pair = output(rel_lines, na_lines, test_filename_list,
           os.path.join(output_path, 'test.txt'),
           None)

    dev_cui_pair = output(rel_lines, na_lines, dev_filename_list,
           os.path.join(output_path, 'dev.txt'),
           test_cui_pair)

    cui_pair = output(rel_lines, na_lines,
           [filename for filename in filename_list if filename not in test_filename_list and filename not in dev_filename_list],
           os.path.join(output_path, 'train.txt'),
           set(test_cui_pair).update(list(dev_cui_pair)))
    # cui_pair = output(rel_lines, na_lines,
    #        filename_list[0:10],
    #        os.path.join(output_path, 'train.txt'),
    #        None)    


if __name__ == '__main__':
    train_test_split(output_path="../dataset_v2")