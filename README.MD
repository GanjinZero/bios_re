# Relation Extraction in BIOS
Project Website: https://bios.idea.edu.cn

BIOS: An Algorithmically Generated Biomedical Knowledge Graph
[Paper](https://arxiv.org/abs/2203.09975)

This repo is under construction.

# Environment
Software dependencies are listed in **requirements.txt**. At least one GPU is needed for training.

# Install guide
Just clone the github repo.

# Train
Use following command to train model.
```
CUDA_VISIBLE_DEVICES=0 python main.py --bag_size 16 --aggr one --batch_size 8 --gradient_accumulation_steps 2 --data_path ./data/1224 --output_base_dir ./output_1224 --limit_dis '3,10'
```

# Predict
Use following command to inference input_data.
```
CUDA_VISIBLE_DEVICES=0 python predict.py input_data output_1224/entity_cls_False_0.1_one_binary_2_2e-05_16_16_0.0_3,10_coder/output.txt
```

input_data is data with json lines, here is a example:
```
{"text": "9-(s)-(2,3-dihydroxypropyl)adenine inhibits the transformation of chick embryo fibroblasts infected with rous sarcoma virus: evidence for inhibition of enzymatic activity of isolated cellular protein kinases by the drug..", "h": {"pos": [66, 78], "id": "CN00468812", "name": "chick embryo"}, "t": {"pos": [105, 123], "id": "CN00449160", "name": "rous sarcoma virus"}, "distance": 5}
{"text": "9-(s)-(2,3-dihydroxypropyl)adenine inhibits the transformation of chick embryo fibroblasts infected with rous sarcoma virus: evidence for inhibition of enzymatic activity of isolated cellular protein kinases by the drug..", "h": {"pos": [66, 78], "id": "CN00468812", "name": "chick embryo"}, "t": {"pos": [138, 148], "id": "CN00129331", "name": "inhibition"}, "distance": 10}
{"text": "9-(s)-(2,3-dihydroxypropyl)adenine inhibits the transformation of chick embryo fibroblasts infected with rous sarcoma virus: evidence for inhibition of enzymatic activity of isolated cellular protein kinases by the drug..", "h": {"pos": [79, 90], "id": "CN00453821", "name": "fibroblasts"}, "t": {"pos": [105, 123], "id": "CN00449160", "name": "rous sarcoma virus"}, "distance": 4}
{"text": "9-(s)-(2,3-dihydroxypropyl)adenine inhibits the transformation of chick embryo fibroblasts infected with rous sarcoma virus: evidence for inhibition of enzymatic activity of isolated cellular protein kinases by the drug..", "h": {"pos": [79, 90], "id": "CN00453821", "name": "fibroblasts"}, "t": {"pos": [138, 148], "id": "CN00129331", "name": "inhibition"}, "distance": 9}
```

Inference with CPU can be slow, a GPU is needed for fast inference.

Example output file can be:
```
{"text": "[CLS] sublethal irradiation with ultraviolet b light appeared to diminish ebv antigen expression ( gp350 / 220 ) during the first 48 to 72 hours in culture , whereas there was no change in the expression of mhc class i or [UNK] immunoglobulin [UNK] host cell [UNK] proteins [UNK] , and an apparent increase in the expression of host cell autoantigens . [SEP]", "h": "immunoglobulin", "t": "proteins", "predict_rel": ["is a"]}
{"text": "[CLS] interestingly , this peptide was recognized by only a few ( less than or equal to 7 % ) of sle sera , while 63 % of pss sera and 46 % of sle sera tested in parallel [UNK] possessed [UNK] antibodies reacting in elisa with purified 60 - kd [UNK] ssa protein [UNK] . [SEP]", "h": "possessed", "t": "ssa protein", "predict_rel": ["inverse involved in"]}
```
