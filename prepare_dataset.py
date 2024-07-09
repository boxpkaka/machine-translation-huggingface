from torch.utils.data import Dataset, IterableDataset
from typing import List, Dict, Iterator
from multiprocessing import Pool
import json
import numpy as np


class TranslationDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length: int = 256):
        self.data = self._load_data(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_lang, tgt_lang = list(item.keys())
        inputs = self.tokenizer(item[src_lang],
                    text_target=item[tgt_lang],
                    max_length=128,
                    truncation=True,
                    padding='max_length',
                    return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs
    
    @staticmethod
    def _load_data(json_path: str, processes=32) -> List[Dict[str, str]]:
        with open(json_path, 'r') as fin:
            f_json = json.load(fin)

        lang_paths = f_json['data_path']
        lang_couples = f_json['lang_couples']  # Dict: {lang:lang_1, lang_2:lang3}
        
        data = []
        for data_path in lang_paths:
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            chunk_size = len(lines) // processes + 1
            chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
            with Pool(processes=processes) as pool:
                chunk_data = pool.map(TranslationDataset._process_chunk, [(chunk, lang_couples) for chunk in chunks])
            for chunk in chunk_data:
                data.extend(chunk)
        return np.array(data)

    @staticmethod
    def _process_chunk(args):
        chunk, lang_couples = args
        data = []
        for line in chunk:
            item = json.loads(line)
            for src_lang, tgt_langs in lang_couples.items():
                for tgt_lang in tgt_langs:
                    try:
                        data.append({src_lang: item[src_lang], tgt_lang: item[tgt_lang]})
                    except KeyError:
                        print(f"Chunk doesn't have the language: {src_lang}/{tgt_lang}")
        return data


class TranslationIterableDataset(IterableDataset):
    def __init__(self, json_path, tokenizer, max_length: int = 256, buffer_size: int = 1000):
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self._prepare_data()

    def _prepare_data(self):
        with open(self.json_path, 'r') as fin:
            f_json = json.load(fin)
        self.lang_paths = f_json['data_path']
        self.lang_couples = f_json['lang_couples']

    def __iter__(self) -> Iterator[Dict[str, Dict[str, List[int]]]]:
        buffer = []
        for data_path in self.lang_paths:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    for src_lang, tgt_langs in self.lang_couples.items():
                        for tgt_lang in tgt_langs:
                            try:
                                data = {src_lang: item[src_lang], tgt_lang: item[tgt_lang]}
                                buffer.append(self._tokenize_data(data))
                                if len(buffer) >= self.buffer_size:
                                    for record in buffer:
                                        yield record
                                    buffer = []
                            except KeyError:
                                print(f"Item doesn't have the language: {src_lang}/{tgt_lang}")
        if buffer:
            for record in buffer:
                yield record

    def _tokenize_data(self, data: Dict[str, str]) -> Dict[str, List[int]]:
        src_lang, tgt_lang = list(data.keys())
        inputs = self.tokenizer(data[src_lang],
                                text_target=data[tgt_lang],
                                max_length=self.max_length,
                                truncation=True,
                                padding='max_length',
                                return_tensors="pt")
        inputs = {k: v.squeeze(0).tolist() for k, v in inputs.items()}
        return inputs


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/opus_zh_en.json", type=str)
    parser.add_argument('--is_nllb', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("/srv/model/huggingface/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("/srv/model/huggingface/opus-mt-zh-en")
    
    train_datasets = TranslationDataset(args.data, tokenizer)
    

