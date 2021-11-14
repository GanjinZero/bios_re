import os
import spacy
import ujson
from load_umls import UMLS
from random import sample

with open('inverse_rel.json', 'r') as f:
    inverse_rel = ujson.load(f)
with open('rel2sty.json', 'r') as f:
    rel2sty = ujson.load(f)

ava_sty = set()
for rel in rel2sty:
    if not rel in inverse_rel:
        ava_sty.update(rel2sty[rel])

umls = UMLS("/media/sda1/GanjinZero/UMLSBert/umls", only_load_dict=True)
umls.load_sty()


def check(ent0, ent1):
    cui0 = ent0['cui']
    cui1 = ent1['cui']
    if cui0 == cui1:
        return False
    if not cui0 in umls.cui2sty:
        return False
    if not cui1 in umls.cui2sty:
        return False
    sty = f'{umls.cui2sty[cui0]}\t{umls.cui2sty[cui1]}'
    if sty in ava_sty:
        return True
    return False


def deal_pubmed_abstract(json_string, nlp, head_dict):
    abstract = json_string['Abstract']
    sentencize = nlp(abstract).to_json()['sents']
    entities = json_string['ents']
    entities = [{'start':ent['start'], 'end':ent['end'], 'cui':ent['umls_link'][0][0]} for ent in entities if ent['umls_link']]
    result = []
    for idx, sen in enumerate(sentencize):
        sen_start_idx = sen['start']
        sen_end_idx = sen['end']
        sen_text = abstract[sen_start_idx:sen_end_idx]
        if sen_text.strip().find('\n') >= 0:
            continue
        sen_entities = [ent for ent in entities if ent['start'] >= sen_start_idx and ent['end'] <= sen_end_idx]
        sample_ent0 = sample(sen_entities, int(0.1 * len(sen_entities)))

        for ent0 in sample_ent0:
            rel_flag = False
            sample_ent1 = sample(sen_entities, int(0.1 * len(sen_entities)))
            if not ent0['cui'] in head_dict:
                for ent1 in sample_ent1:
                    if not check(ent0, ent1):
                        continue
                    d = {'text':sen_text, 'relation':'NA', 
                        'h':{'pos':[ent0['start'] - sen_start_idx, ent0['end'] - sen_start_idx], 'id':ent0['cui'], 'name':abstract[ent0['start']:ent0['end']]},
                        't':{'pos':[ent1['start'] - sen_start_idx, ent1['end'] - sen_start_idx], 'id':ent1['cui'], 'name':abstract[ent1['start']:ent1['end']]}}
                    result.append(d)
            else:
                for ent1 in sample_ent1:
                    if not ent1['cui'] in head_dict[ent0['cui']]:
                        if not ent1['cui'] in head_dict or (not ent0['cui'] in head_dict[ent1['cui']]):
                            if not check(ent0, ent1):
                                continue
                            d = {'text':sen_text, 'relation':'NA', 
                                'h':{'pos':[ent0['start'] - sen_start_idx, ent0['end'] - sen_start_idx], 'id':ent0['cui'], 'name':abstract[ent0['start']:ent0['end']]},
                                't':{'pos':[ent1['start'] - sen_start_idx, ent1['end'] - sen_start_idx], 'id':ent1['cui'], 'name':abstract[ent1['start']:ent1['end']]}}
                            result.append(d)
    return result


def deal_pubmed_file(filename, nlp, head_dict):
    print(filename)
    res_count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'Abstract count: {len(lines)}')
    for line in lines:
        result = deal_pubmed_abstract(eval(line.strip()), nlp, head_dict)
        res_count += len(result)
        if res_count > 0.1 * len(lines):
            break
        with open(os.path.join("../na_rel", os.path.basename(filename)), 'a+', encoding='utf-8') as f:
            for res in result:
                f.write(f'{ujson.dumps(res)}\n')
    print(f'Sentences count: {res_count}')
    

def main():
    nlp = spacy.load("/media/sdb1/ZhengyunZhao/REDataset/scibert", exclude = ["attribute_ruler", "lemmatizer", "transformer", "ner"])
    nlp.add_pipe('sentencizer')
    # NA relation cannot appear in all sources
    with open('full/head_dict.json', 'r') as f:
        head_dict = ujson.load(f)
    for filename in os.listdir('/media/sdb1/ZhengyunZhao/REDataset/Pubmed_Abstract_scispacy_NER/'):
        if not os.path.exists(os.path.join('../na_rel', filename)):
            try:
                deal_pubmed_file(os.path.join('/media/sdb1/ZhengyunZhao/REDataset/Pubmed_Abstract_scispacy_NER/', filename), nlp, head_dict)
            except BaseException:
                pass


if __name__ == '__main__':
    main()
