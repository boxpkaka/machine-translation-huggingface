import sentencepiece as spm
import sys
import os


def get_combined_vocab(spm_path_1, spm_path_2):
    sp_1 = spm.SentencePieceProcessor()
    sp_1.load(spm_path_1)

    sp_2 = spm.SentencePieceProcessor()
    sp_2.load(spm_path_2)

    vocab_1 = set(sp_1.id_to_piece(id) for id in range(sp_1.get_piece_size()))
    vocab_2 = set(sp_2.id_to_piece(id) for id in range(sp_2.get_piece_size()))

    vocab = vocab_1 | vocab_2
    
    print(vocab)
    print(len(vocab))


if __name__ == "__main__":
    spm_path_1 = sys.argv[1]
    spm_path_2 = sys.argv[2]
    get_combined_vocab(spm_path_1, spm_path_2)

    
