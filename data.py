import torch

from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k 

from config import batch_size

"""
ref : https://tutorials.pytorch.kr/beginner/torchtext_translation_tutorial.html
"""

"""
python -m spacy download en
python -m spacy download de
"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

src = Field(tokenize = "spacy",
            tokenizer_language="de_core_news_sm",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

trg = Field(tokenize = "spacy",
            tokenizer_language="en_core_web_sm",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (src, trg))

src.build_vocab(train_data, min_freq = 2)
trg.build_vocab(train_data, min_freq = 2)

train_iter, valid_iter, test_iter = BucketIterator.splits((train_data, valid_data, test_data),
                                                          batch_size = batch_size,
                                                          device = device)

enc_voc_size = len(src.vocab)
dec_voc_size = len(trg.vocab)

pad_idx = src.vocab.stoi['<pad>']

def idx_to_word(x, vocab):
    words = []
    for i in x:
        word = vocab.itos[i]
        words.append(word)
    return words