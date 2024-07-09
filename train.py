from prepare_dataset import TranslationDataset, TranslationIterableDataset
from tensorboard_cpu_memo import MemoryMonitorCallback

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse

gradient_checkpoint = True
use_8bit = True


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, local_files_only=True)
    if gradient_checkpoint:
        model.gradient_checkpointing_enable()
    
    train_dataset = TranslationDataset(args.train_data, tokenizer)
    val_dataset = TranslationDataset(args.val_data, tokenizer)
 
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=20000,
        save_steps=500,
        save_total_limit=5,
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        weight_decay=0.01,
        num_train_epochs=10,
        predict_with_generate=True,
        gradient_checkpointing=False,
        bf16=True,
        report_to=["tensorboard"],
        logging_dir=args.logging_dir,
        logging_steps=100,
        dataloader_num_workers=16,
        # max_steps=1126126
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[MemoryMonitorCallback(log_dir=args.logging_dir)]
    )
    if args.checkpoint:
        trainer.train(resume_from_checkpoint=args.checkpoint)
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval whisper')
    parser.add_argument('--train_data',             help='JSON path of dataset',       type=str)
    parser.add_argument('--val_data',               help='JSON path of dataset',       type=str)
    parser.add_argument('--model_dir',              help='model directory',            type=str)
    parser.add_argument('--is_nllb',                action='store_true')
    parser.add_argument('--train_batch',             type=int)
    parser.add_argument('--eval_batch',              type=int)
    parser.add_argument('--checkpoint',              default=None,                     type=str)
    parser.add_argument('--logging_dir',             default=None,                     type=str)
    parser.add_argument('--output_dir',              default=None,                     type=str)
    
    args = parser.parse_args()
    main(args)
