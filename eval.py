from prepare_dataset import TranslationDataset

from typing import List
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from comet.models import load_from_checkpoint
from transformers import XLMRobertaTokenizerFast
from pythainlp.tokenize import word_tokenize
from pecab import PeCab

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sacrebleu
import argparse
import textwrap
import torch
import MeCab
import jieba
import json
import os
import re

def smooth(data: np.array, window_size: int = 100) -> np.array:
    """
    对输入数据应用移动平均平滑处理。

    参数:
    data (np.array): 输入的NumPy数组数据。
    window_size (int): 移动平均的窗口大小，默认为100。

    返回:
    np.array: 平滑后的数据。
    """
    series = pd.Series(data)
    moving_average = series.rolling(window=window_size).mean()
    smoothed_data = moving_average.to_numpy()
    return smoothed_data

def tokenize(sentence: str, lang: str) -> str:
    if lang not in {"zh-cn", "ja", "ko", "th"}:
        return sentence
    
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = sentence.replace(' ', '')
    if lang == "ja":
        tagger = MeCab.Tagger("-Owakati")
        return tagger.parse(sentence).strip()
    if lang == 'zh-cn':
        tmp = jieba.cut(sentence)
        return ' '.join(tmp).strip()
    if lang == 'ko':
        pecab = PeCab()
        tmp = pecab.morphs(sentence)
        return ' '.join(tmp).strip()
    if lang == 'th':
        tmp = word_tokenize(sentence, engine='newmm')
        return ' '.join(tmp).strip()

def draw_eval_data(data: List[List[float]], 
                   title: List[str], 
                   window_sizes: List[int], 
                   save_dir: str) -> None:
    """
    绘制保存测试数据。

    参数:
    data (List[List[float]]): 输入的数据，其中每个List中为一组数据。
    title: List[str]：每组数据的标题。
    window_sizes (List[int]): 每组数据的一组平滑窗口大小。
    save_dir: str：保存目录

    返回:
    np.array: 平滑后的数据。
    """
    assert len(data) == len(title)
    fig, axs = plt.subplots(2, 1, figsize=(8, 12))
    
    plt.legend()
    metrics = []
    for i in range(len(data)):
        metrics.append((title[i], data[i]))

    for i, (title, metric) in enumerate(metrics):
        axs[i].plot(metric, label='Original')
        for window_size in window_sizes:
            smoothed_metric = smooth(metric, window_size)
            axs[i].plot(smoothed_metric, label=f'Smoothed (window={window_size})')
        axs[i].set_title(title)
        axs[i].legend()
    
    plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.savefig(os.path.join(save_dir, 'eval.png'))
      
def _infer_save(args):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
        
    eval_datasets = TranslationDataset(args.data, tokenizer)
    dataloader = DataLoader(dataset=eval_datasets,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
    
    with open(args.data, 'r') as fin:
        f_json = json.load(fin)
        tgt_lang = list(f_json["lang_couples"].values())[0][0]

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'ref'), 'w', encoding='utf-8') as ref, \
        open(os.path.join(args.save_dir, 'tra'), 'w', encoding='utf-8') as tra, \
        open(os.path.join(args.save_dir, 'orig'), 'w', encoding='utf-8') as orig:
        for batch in tqdm(dataloader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model.generate(inputs, num_beams=4)
            _translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            _labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            _inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
            
            for translation, label, origin in zip(_translations, _labels, _inputs):
                # print(translation)
                label = tokenize(label, tgt_lang)
                translation = tokenize(translation, tgt_lang)
                # print(translation)
                tra.write(translation + '\n')
                ref.write(label + '\n')
                orig.write(origin + '\n')
                
def _eval_with_metric(args):
    result = []
    trans = []
    labels = []
    torch.set_float32_matmul_precision('high')
    model = load_from_checkpoint('/srv/model/comet/checkpoints/model.ckpt').to(args.device)
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('/srv/model/huggingface/xlm-roberta-large')
    model.tokenizer =tokenizer
    
    comet_data = []
    with open(os.path.join(args.save_dir, 'ref'), 'r', encoding='utf-8') as refs, \
        open(os.path.join(args.save_dir, 'tra'), 'r', encoding='utf-8') as tras, \
        open(os.path.join(args.save_dir, 'orig'), 'r', encoding='utf-8') as orig, \
        open(os.path.join(args.save_dir, 'sentence_score'), 'w', encoding='utf-8') as sentence_score, \
        open(os.path.join(args.save_dir, 'corpus_score'), 'w', encoding='utf-8') as corpus_score:
        for label, translation, origin in zip(refs, tras, orig):
            label, translation = label.strip(), translation.strip()
            bleu = sacrebleu.sentence_bleu(translation, [label])
            result.append([bleu.score, translation, label, origin.strip()])
            comet_data.append({"src": origin, "mt": translation, "ref": label})
        
        outputs = model.predict(comet_data, batch_size=16, gpus=1)
        comet_scores = outputs['scores']
        comet_score_all = outputs['system_score']
        for i in range(len(result)):
            result[i].append(comet_scores[i])
        sorted_result = sorted(result, key=lambda x: x[0], reverse=False)
        plot_data = list(zip(*sorted_result))
        draw_eval_data(
            data=[plot_data[0], plot_data[4]],
            title=['bleu', 'comet'],
            window_sizes=[10, 50, 100],
            save_dir=args.save_dir,
        )
        for bleu_score, translation, label, origin, comet_score in sorted_result:
            trans.append(translation)
            labels.append([label])
            sentence_score.write(textwrap.dedent(f"""
                                **** 
                                BLEU:        {bleu_score:.4f} 
                                COMET:       {comet_score:.4f}
                                Source:      {origin} 
                                Translation: {translation} 
                                Reference:   {label}
                                """))

        all_bleu =  corpus_bleu(labels, trans)
        corpus_score.write(f"BLEU:  {all_bleu * 100:.4f} \n")
        corpus_score.write(f"COMET: {comet_score_all:.4f}")

def _eval(args):
    if args.only_eval:
        _eval_with_metric(args)
    else:
        _infer_save(args)
        _eval_with_metric(args)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',            type=str)
    parser.add_argument('--model',           type=str)
    parser.add_argument('--save_dir',        type=str, default='eval/')
    parser.add_argument('--batch_size',      type=int, default=32)
    parser.add_argument('--num_workers',     type=int, default=16)
    parser.add_argument('--device',          type=str, default="cuda:1")
    parser.add_argument('--only_eval',       action='store_true')
    args = parser.parse_args()
    
    _eval(args)
        
    
    
