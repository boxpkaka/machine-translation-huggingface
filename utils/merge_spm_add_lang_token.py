import sentencepiece as spm
import os


def get_combined_vocab(model_dir):
    model_dir = '/srv/model/huggingface/opus-mt-zh-en'
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(os.path.join(model_dir,'target.spm'))

    sp_zh = spm.SentencePieceProcessor()
    sp_zh.load(os.path.join(model_dir,'source.spm'))

    vocab_en = [sp_en.id_to_piece(id) for id in range(sp_en.get_piece_size())]
    vocab_zh = [sp_zh.id_to_piece(id) for id in range(sp_zh.get_piece_size())]

    # 去重并合并词汇表，中文词汇添加在英文词汇之后
    combined_vocab = vocab_en + [token for token in vocab_zh if token not in vocab_en]

    # 添加特殊的语言标识符
    # special_tokens = ['<en>', '<zh>']
    # combined_vocab = special_tokens + combined_vocab
    # print(combined_vocab[:100])
    # 将合并的词汇表写入文件
    with open('merge_spm/combined_vocab.txt', 'w', encoding='utf-8') as f:
        for token in combined_vocab:
            f.write(f'{token}\n')
    
    return combined_vocab

def train_spm_model(model_dir):
    combined_vocab = get_combined_vocab(model_dir)
    spm.SentencePieceTrainer.train(input='merge_spm/combined_vocab.txt', 
                                   model_prefix='merge_spm/combined_model',
                                   vocab_size=len(combined_vocab), 
                                   model_type='bpe',
                                   accept_language='zh,en')

if __name__ == "__main__":
    # get_combined_vocab('/srv/model/huggingface/opus-mt-zh-en')
    train_spm_model('/srv/model/huggingface/opus-mt-zh-en')
