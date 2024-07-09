from prepare_dataset import TranslationDataset

from comet import download_model, load_from_checkpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sentencepiece as spm
import sacrebleu
import argparse
import torch
import os


def _infer_save(args):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    
    eval_datasets = TranslationDataset(args.data, tokenizer)
    dataloader = DataLoader(dataset=eval_datasets,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
    
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'ref'), 'w', encoding='utf-8') as ref, \
        open(os.path.join(args.save_dir, 'tra'), 'w', encoding='utf-8') as tra, \
        open(os.path.join(args.save_dir, 'orig'), 'w', encoding='utf-8') as orig:
        for batch in tqdm(dataloader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model.generate(inputs)
            _translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            _labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            _inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

            for translation, label, origin in zip(_translations, _labels, _inputs):
                tra.write(translation + '\n')
                ref.write(label + '\n')
                orig.write(origin + '\n')
                
                
def _eval_with_bleu(args):
    result = []
    trans = []
    labels = []
    with open(os.path.join(args.save_dir, 'ref'), 'r', encoding='utf-8') as refs, \
        open(os.path.join(args.save_dir, 'tra'), 'r', encoding='utf-8') as tras, \
        open(os.path.join(args.save_dir, 'orig'), 'r', encoding='utf-8') as orig, \
        open(os.path.join(args.save_dir, 'sentence_score'), 'w', encoding='utf-8') as sentence_score, \
        open(os.path.join(args.save_dir, 'corpus_score'), 'w', encoding='utf-8') as corpus_score:
        for label, translation, orgin in zip(refs, tras, orig):
            label, translation = label.strip(), translation.strip()
            bleu = sacrebleu.sentence_bleu(translation, [label])
            result.append([bleu, translation, label, orgin.strip()])
        sorted_result = sorted(result, key=lambda x: x[0].score, reverse=True)

        for item in sorted_result:
            trans.append(item[1])
            labels.append([item[2]])
            sentence_score.write(f"{item[0]} | {item[1]} | {item[2]} | {item[3]} \n")

        all_bleu =  sacrebleu.corpus_bleu(labels, trans)
        print(all_bleu)
        corpus_score.write(f"BLEU: {all_bleu.score}")

def _eval_with_comet():

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)
data = [
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    }
]
model_output = model.predict(data, batch_size=8, gpus=1)
print (model_output)



def _eval(args):
    # _infer_save(args)
    _eval_with_metric(args)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',                  type=str)
    parser.add_argument('--model',                 type=str)
    parser.add_argument('--save_dir',     default='eval/',         type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--device',     type=str, default="cuda:0")
    args = parser.parse_args()
    
    _eval(args)
        
    
    
