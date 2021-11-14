import os
import spacy
import ujson

# {"text": "Prato , Italy , is another of Mr. Kynge 's destinations .", 
# "relation": "/location/location/contains", 
# "t": {"pos": [0, 5], "id": "m.05znn0", "name": "Prato"}, 
# "h": {"pos": [8, 13], "id": "m.03rjj", "name": "Italy"}}
with open('inverse_rel.json', 'r') as f:
    inverse_rel = ujson.load(f)


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
        sen_entities = [ent for ent in entities if ent['start'] >= sen_start_idx and ent['end'] <= sen_end_idx]

        for ent0 in sen_entities:
            if not ent0['cui'] in head_dict:
                continue
            for ent1 in sen_entities:
                if ent1['cui'] in head_dict[ent0['cui']]:
                    for rel in head_dict[ent0['cui']][ent1['cui']]:
                        if not rel.split('\t')[1] in inverse_rel:
                            d = {'text':sen_text, 'relation':rel, 
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
        with open(os.path.join("../rxnorm_snomedctus_medrt_rel", os.path.basename(filename)), 'a+', encoding='utf-8') as f:
            for res in result:
                f.write(f'{ujson.dumps(res)}\n')
    print(f'Sentences count: {res_count}')
    

def main():
    nlp = spacy.load("/media/sdb1/ZhengyunZhao/REDataset/scibert", exclude = ["attribute_ruler", "lemmatizer", "transformer", "ner"])
    nlp.add_pipe('sentencizer')
    with open('head_dict.json', 'r') as f:
        head_dict = ujson.load(f)
    for filename in os.listdir('/media/sdb1/ZhengyunZhao/REDataset/Pubmed_Abstract_scispacy_NER/'):
        if not os.path.exists(os.path.join('../rxnorm_snomedctus_medrt_rel', filename)):
            try:
                deal_pubmed_file(os.path.join('/media/sdb1/ZhengyunZhao/REDataset/Pubmed_Abstract_scispacy_NER/', filename), nlp, head_dict)
            except BaseException:
                pass


if __name__ == '__main__':
    main()
