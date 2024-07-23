from prepare_dataset import TranslationDataset, ShardTranslationDataset, IterableTranslationDataset

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import MarianMTModel, MarianConfig
from loguru import logger
import argparse

gradient_checkpoint = True


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    
    if args.init_model:
        config = MarianConfig.from_pretrained(args.model_dir)
        model = MarianMTModel(config)
        logger.info(f"Init model, use config: \n{config}")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, local_files_only=True)
        logger.info(f"Load model from {args.model_dir}. ")
        
    if gradient_checkpoint:
            model.gradient_checkpointing_enable()
            logger.info('Use gradient checkpoint. ')
            # model.config.use_cache = False
    
    train_dataset = IterableTranslationDataset(args.train_data, tokenizer)
    val_dataset = TranslationDataset(args.val_data, tokenizer)

    max_steps = None
    if isinstance(train_dataset, IterableTranslationDataset):
        logger.info(f"Use iterable dataset, counting the max steps. ")
        max_steps = train_dataset.get_max_steps(epochs=args.num_epoch,
                                                num_gpus=args.num_gpu,
                                                batch_size=args.train_batch)
        logger.info(f"Max steps of training: {max_steps}")
    
    train_kwargs = {
        'output_dir': args.output_dir,
        'eval_strategy': "steps",
        'save_strategy': "steps",
        'eval_steps': args.eval_steps,
        'save_steps': args.save_steps,
        'save_total_limit': 5,
        'learning_rate': 2e-5,
        'per_device_train_batch_size': args.train_batch,
        'per_device_eval_batch_size': args.eval_batch,
        'weight_decay': 0.01,
        'num_train_epochs': args.num_epoch,
        'predict_with_generate': True,
        'gradient_checkpointing': gradient_checkpoint,
        'bf16': True,
        'report_to': ["tensorboard"],
        'logging_dir': args.logging_dir,
        'logging_steps': 50,
        'dataloader_num_workers': 16,
    }
    if max_steps:
        train_kwargs['max_steps'] = max_steps
    
    training_args = Seq2SeqTrainingArguments(**train_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    if args.checkpoint:
        logger.info('Resume training from checkpoint: {args.checkpoint}. ')
        trainer.train(resume_from_checkpoint=args.checkpoint)
    else:
        logger.info('Starting training from scratch. ')
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval whisper')
    parser.add_argument('--train_data', help='JSON path of dataset', type=str)
    parser.add_argument('--val_data',    help='JSON path of dataset', type=str)
    parser.add_argument('--model_dir',   help='model directory', type=str)
    parser.add_argument('--is_nllb',        action='store_true')
    parser.add_argument('--init_model',     action='store_true')
    parser.add_argument('--train_batch',       type=int)
    parser.add_argument('--eval_batch',        type=int)
    parser.add_argument('--num_epoch',         type=int)
    parser.add_argument('--num_gpu',           type=int)
    parser.add_argument('--eval_steps',        type=int)
    parser.add_argument('--save_steps',        type=int)
    parser.add_argument('--checkpoint',    default=None, type=str)
    parser.add_argument('--logging_dir',   default=None, type=str)
    parser.add_argument('--output_dir',    default=None, type=str)
    
    args = parser.parse_args()
    main(args)
