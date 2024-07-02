from datasets import Dataset, concatenate_datasets
from typing import List, Dict
from functools import partial

import torch
import json


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data:List[Dict[str, str]], json_path: str ,max_length=128, truncation=True, padding=True ) -> None:
        self.data = data
        
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pass
        
    

def get_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        file = [line.strip() for line in f]
    return file

def _get_raw_dataset(json_path: str):
    with open(json_path, 'r') as fin:
        f_json = json.load(fin)

    language_data = {}
    lang_path_list = f_json['data_path']
    lang_couples = f_json['lang_couples']   # Dict: {lang:[lang_1, lang_2, ...], ...}
    for item in lang_path_list:
        language = item['language']
        paths = item['path']
        all_lines = []
        for path in paths:
            lines = get_lines(path)
            all_lines = all_lines + lines
            
        language_data[language] = all_lines
    line_counts = [len(lines) for lines in language_data.values()]
    assert len(set(line_counts)) == 1, "行数不一致"
    
    languages = list(language_data.keys())

    data = {"translation": []}
    for i in range(line_counts[0]):
        translation_entry = {lang: language_data[lang][i] for lang in languages}
        data["translation"].append(translation_entry)
   
    dataset = Dataset.from_dict(data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']

    return train_dataset, val_dataset, lang_couples

def _preprocess_function(dataset, src_lang, tgt_lang, tokenizer, is_nllb):
    if is_nllb:
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

    inputs = [item[src_lang] for item in dataset["translation"]]
    targets = [item[tgt_lang] for item in dataset["translation"]]
    tokenized_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True, padding=True)

    return tokenized_inputs

def get_mapped_dataset(args, tokenizer):
    train_dataset, val_dataset, lang_couples = _get_raw_dataset(args.data)
    
    final_train_dataset = []
    final_val_dataset = []
    for src_lang, tgt_lang_list in lang_couples.items():
        for tgt_lang in tgt_lang_list:
            preprocess = partial(_preprocess_function, 
                                src_lang=src_lang,
                                tgt_lang=tgt_lang,
                                tokenizer=tokenizer,
                                is_nllb=args.is_nllb)
            map_kwargs={
                "function": preprocess, 
                "batched": True,
                "batch_size": 5000,
                "writer_batch_size": 5000,
                "num_proc": 16,
                "load_from_cache_file": True}
            tokenized_train_dataset = train_dataset.map(**map_kwargs)
            tokenized_val_dataset = val_dataset.map(**map_kwargs) 

            final_train_dataset.append(tokenized_train_dataset)
            final_val_dataset.append(tokenized_val_dataset)
            
    final_train_dataset = concatenate_datasets(final_train_dataset)
    final_val_dataset = concatenate_datasets(final_val_dataset)
            
    return final_train_dataset, final_val_dataset 


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/opus_zh_es.json",       type=str)
    parser.add_argument('--is_nllb', action='store_true')
    tokenizer = AutoTokenizer.from_pretrained("/srv/model/huggingface/opus-mt-zh-en")
    args = parser.parse_args()
    
    _, _ = get_mapped_dataset(args, tokenizer)
    

