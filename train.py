from prepare_dataset import get_mapped_dataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse

gradient_checkpoint = True
use_8bit = True


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    if gradient_checkpoint:
        model.gradient_checkpointing_enable()
    
    train_dataset, val_dataset = get_mapped_dataset(args.data, tokenizer)
    print(len(train_dataset))
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        num_train_epochs=2,
        predict_with_generate=True,
        gradient_checkpointing=True,
        bf16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval whisper')
    parser.add_argument('--data',                   help='JSON path of dataset',       type=str)
    parser.add_argument('--model_dir',              help='model directory',              type=str)
    
    # parser.add_argument('--model_index',            help='index of model list',        type=int)
    # parser.add_argument('--lora_dir', default=None, help='directory of LoRA file',     type=str)
    # parser.add_argument('--data_index',             help='index of dataset list',      type=int)
    # parser.add_argument('--language',               help='whisper inference language', type=str)
    # parser.add_argument('--batch_size',             help='batch size',                 type=int)
    # parser.add_argument('--use_cpu', default=False, help='ct2: use cpu of ct2model',   type=bool)
    # parser.add_argument('--compute_type',           help='ct2: auto/int8/float16...',  type=str)
    # parser.add_argument('--num_workers',            help='num of workers          ',   type=int)
    # parser.add_argument('--pipeline',               help='use transformers pipeline',  type=int)
    # parser.add_argument('--use_flash_attention_2',  help='whether use flash attn 2',   type=int)
    # parser.add_argument('--torch_dtype',            help='fp16, bf16',                 type=str)
    # parser.add_argument('--use_bettertransformer',  help='pipeline options',           type=int)
    # parser.add_argument('--use_compile',            help='pipeline options',           type=int)
    # parser.add_argument('--assistant_model_path',   help='pipeline options',           type=str)
    # parser.add_argument('--preheat',                help='whether preheat first',      type=int)
    # parser.add_argument('--gpu',     default=0,     help='gpu id',                     type=str)
    args = parser.parse_args()
    main(args)
