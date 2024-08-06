from torch.utils.data import Dataset, IterableDataset
from typing import List, Dict, Iterator
from multiprocessing import Pool
from loguru import logger
import ujson
import os
import torch
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
            f_json = ujson.load(fin)

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
            item = ujson.loads(line)
            src_lang = next(iter(lang_couples))
            tgt_lang = lang_couples[src_lang][0]
            if src_lang not in item or tgt_lang not in item:
                continue
            try:
                data.append({src_lang: item[src_lang], tgt_lang: item[tgt_lang]})
            except KeyError:
                print(f"Doesn't have the language: {src_lang}/{tgt_lang}")
        return data

class ShardTranslationDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length: int = 256):
        self.json_files = self._init_get_data(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.file_lengths, self.total_length = self._compute_file_lengths()
        self.current_file_idx = -1 
        self.data = []
        self.current_min_idx = 0
        self.current_max_idx = -1

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if not (self.current_min_idx <= idx <= self.current_max_idx):
            self._load_data_for_idx(idx)
        local_idx = idx - self.current_min_idx
        item = self.data[local_idx]
        src_lang, tgt_lang = list(item.keys())
        
        inputs = self.tokenizer(item[src_lang],
                                text_target=item[tgt_lang],
                                max_length=self.max_length,
                                truncation=True,
                                padding='max_length',
                                return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

    def _compute_file_lengths(self):
        with open(self.json_files[0], 'r', encoding='utf-8') as f:
            first_file_data = ujson.load(f)
            first_file_length = len(first_file_data)
        
        num_files = len(self.json_files)
        file_lengths = [first_file_length] * (num_files - 1)

        with open(self.json_files[-1], 'r', encoding='utf-8') as f:
            last_file_data = ujson.load(f)
            last_file_length = len(last_file_data)

        file_lengths.append(last_file_length)
        total_length = first_file_length * (num_files - 1) + last_file_length
        
        return file_lengths, total_length

    def _load_data_for_idx(self, idx: int):
        total_length = 0
        for i, length in enumerate(self.file_lengths):
            if total_length <= idx < total_length + length:
                self.current_file_idx = i
                self.data = self._load_data_from_file(i)
                self.current_min_idx = total_length
                self.current_max_idx = total_length + length - 1
                break
            total_length += length

    def _load_data_from_file(self, file_idx: int) -> List[Dict[str, str]]:
        with open(self.json_files[file_idx], 'r', encoding='utf-8') as f:
            return ujson.load(f)
        
    def _init_get_data(self, json_path: str):
        with open(json_path, 'r') as fin:
            f_json = ujson.load(fin)

        shard_dir = f_json['data_path'][0]
        
        json_files = [os.path.join(shard_dir, name) for name in os.listdir(shard_dir)]
        json_files.sort(key=lambda x:int(x.split('/')[-1].split('.json')[0]))
        
        return json_files


class IterableTranslationDataset(IterableDataset):
    def __init__(self, json_path, tokenizer, max_length: int = 256, buffer_size: int = 1000000):
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.data_paths = None
        self.lang_couples = None
        self._prepare_data()

    def _prepare_data(self):
        with open(self.json_path, 'r') as fin:
            f_json = ujson.load(fin)
        self.data_paths = f_json['data_path']
        self.lang_couples = f_json['lang_couples']

    def __iter__(self) -> Iterator[Dict[str, Dict[str, List[int]]]]:
        buffer = []
        for data_path in self.data_paths:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = ujson.loads(line)
                    except ujson.JSONDecodeError as e:
                        continue
                    src_lang = next(iter(self.lang_couples))
                    tgt_lang = self.lang_couples[src_lang][0]
                    if src_lang not in item or tgt_lang not in item:
                        continue
                    if item[src_lang] != "" and item[tgt_lang] != "":
                        corpus = {src_lang: item[src_lang], tgt_lang: item[tgt_lang]}
                        data = self._tokenize_data(corpus, src_lang, tgt_lang)
                        assert isinstance(data['input_ids'], torch.Tensor)
                        assert isinstance(data['labels'], torch.Tensor)
                        buffer.append(data)
                        if len(buffer) >= self.buffer_size:
                            yield from buffer
                            buffer = []
                    else:
                        continue
        if buffer:
            yield from buffer

    def _tokenize_data(self, data: Dict[str, str], src_lang: str, tgt_lang: str) -> Dict[str, List[int]]:
        inputs = self.tokenizer(data[src_lang],
                                text_target=data[tgt_lang],
                                max_length=self.max_length,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt')
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

    def _count_samples(self) -> int:
        count = 0
        src_lang = list(self.lang_couples.keys())[0]
        tgt_lang = self.lang_couples[src_lang][0]
        
        if os.path.exists(self.data_paths[0] + '.num'):
            # Use the num file if it exists
            num_file_path = self.data_paths[0] + '.num'
            logger.info(f"Use {num_file_path} instead of counting!")
            with open(num_file_path, 'r') as f_cout:
                count_dict = ujson.load(f_cout)
                count = min(count_dict[src_lang], count_dict[tgt_lang])
        else:
            for data_path in self.lang_paths:
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = ujson.loads(line)
                        except ujson.JSONDecodeError as e:
                            continue
                        if src_lang not in item or tgt_lang not in item:
                            continue
                        if item[src_lang] != "" and item[tgt_lang] != "":
                            count += 1
        logger.info(f"Number of dataset {count}")
        return count
    
    def get_max_steps(self, epochs: int, num_gpus: int, batch_size: int) -> int:
        total_samples = self._count_samples()
        total_batches_per_epoch = total_samples // (num_gpus * batch_size)
        return total_batches_per_epoch * epochs


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import argparse
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/opus_en_ko.json", type=str)
    parser.add_argument('--is_nllb', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("/srv/model/huggingface/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("/srv/model/huggingface/opus-mt-zh-en")
    
    train_datasets = IterableTranslationDataset(args.data, tokenizer)
    dataloader = DataLoader(dataset=train_datasets, batch_size=2048, num_workers=16)
    
    for batch in tqdm(dataloader):
        print(batch)
        break


