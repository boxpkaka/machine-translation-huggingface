import sentencepiece as spm
import sys
import os


def train_spm(text_file: str, language: str, vocab_size: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model_prefix = os.path.join(save_dir, f"{text_file.split('/')[-1]}-{language}")
    print('model save in:', model_prefix)
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='unigram',
        character_coverage=1,  
        max_sentencepiece_length=6,
        byte_fallback=True,
        unk_id=0,  
        bos_id=1,  
        eos_id=2   
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')

    text = "地元 メディア の 報道 に よる と 空港 の 消防 車 が 対応 中 に 横転 し た と いう こと です"
    pieces = sp.encode_as_pieces(text)
    ids = sp.encode_as_ids(text)

    print(f"Text: {text}")
    print(f"Pieces: {pieces}")
    print(f"Ids: {ids}")
    
    
if __name__ == "__main__":
    text_file = sys.argv[1]
    language = sys.argv[2]
    vocab_size = sys.argv[3]
    save_dir = sys.argv[4]
    train_spm(text_file, language, int(vocab_size), save_dir)
