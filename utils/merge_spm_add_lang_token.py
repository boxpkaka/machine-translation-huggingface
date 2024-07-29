import sentencepiece as spm
import json
import sys
import os


def get_combined_vocab(spm_path_1: str, spm_path_2: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    sp_1 = spm.SentencePieceProcessor()
    sp_1.load(spm_path_1)

    sp_2 = spm.SentencePieceProcessor()
    sp_2.load(spm_path_2)

    vocab_1_set = {sp_1.id_to_piece(id) for id in range(sp_1.get_piece_size())}
    vocab_2_set = {sp_2.id_to_piece(id) for id in range(sp_1.get_piece_size())}
    share_vocab_set = vocab_1_set | vocab_2_set
    
    vocab_1 = {sp_1.id_to_piece(id): id for id in range(sp_1.get_piece_size())}
    vocab_2 = {sp_2.id_to_piece(id): id for id in range(sp_2.get_piece_size())}

    combined_vocab = {}
    cnt = 0
    for k in vocab_1:
        if k in share_vocab_set:
            combined_vocab[k] = cnt
            share_vocab_set.remove(k)
            cnt += 1
    
    for k in vocab_2:
        if k in share_vocab_set:
            combined_vocab[k] = cnt
            share_vocab_set.remove(k)
            cnt += 1

    with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as fout:
        json.dump(combined_vocab, fout, ensure_ascii=False)


if __name__ == "__main__":
    spm_path_1 = sys.argv[1]
    spm_path_2 = sys.argv[2]
    save_dir = sys.argv[3]
    get_combined_vocab(spm_path_1, spm_path_2, save_dir)

    
