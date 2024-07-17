from prepare_dataset import TranslationDataset

from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from comet.models import load_from_checkpoint
from transformers import XLMRobertaTokenizerFast
from pythainlp.tokenize import word_tokenize
from pecab import PeCab

import sentencepiece as spm
import sacrebleu
import argparse
import textwrap
import torch
import MeCab
import jieba
import json
import os
import re


def tokenize(sentence: str, lang: str) -> str:
    if lang not in ["zh-cn", "ja", "ko", "th"]:
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


def _infer_zhenhui(args):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    with open(args.data, 'r') as f_json:
        _json = json.load(f_json)
        data_path = _json['data_path'][0]
        lang_couples = _json['lang_couples']
        src_lang = None
        tgt_lang = None
        for k, v in lang_couples.items():
            src_lang = k
            tgt_lang = v[0]
            break
    
    print(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(data_path, 'r', encoding='utf-8') as fin, open(os.path.join(args.save_dir, 'ref'), 'w', encoding='utf-8') as ref, \
        open(os.path.join(args.save_dir, 'tra'), 'w', encoding='utf-8') as tra, \
        open(os.path.join(args.save_dir, 'orig'), 'w', encoding='utf-8') as orig:
        for line in tqdm(fin):
            lang_text_dict = json.loads(line)
            origin = lang_text_dict[src_lang]
            label = lang_text_dict[tgt_lang]
            inputs = tokenizer(origin,return_tensors="pt",).to(device)
            outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            tra.write(translation + '\n')
            ref.write(label + '\n')
            orig.write(origin + '\n')
            
            
            
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
                label = tokenize(label, tgt_lang)
                translation = tokenize(translation, tgt_lang)
                
                tra.write(translation + '\n')
                ref.write(label + '\n')
                orig.write(origin + '\n')
                
def _eval_with_metric(args):
    result = []
    trans = []
    labels = []
    torch.set_float32_matmul_precision('high')
    model = load_from_checkpoint('/srv/model/comet/checkpoints/model.ckpt')
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
        sorted_result = sorted(result, key=lambda x: x[0], reverse=True)
            
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
    _infer_save(args)
    # _infer_zhenhui(args)
    _eval_with_metric(args)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',                  type=str)
    parser.add_argument('--model',                 type=str)
    parser.add_argument('--save_dir',     default='eval/',         type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device',     type=str, default="cuda:1")
    args = parser.parse_args()
    
    _eval(args)
        
    
    
