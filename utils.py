#-*-coding:utf8-*-

__author="buyizhiyou"
__date = "2018-7-30"

import pdb
import re
from collections import Counter
import matplotlib.pyplot as plt

def tokenizer(sentence):
    tokens = re.findall(r"[\w]+|[^\s\w]",sentence)
    return tokens

# print(tokenizer('Hello world?? "sdfs%@#%'))

def build_vocab(sentences,is_target=False,max_vocab_size=None):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()

    for sentence in sentences:
        tokens = tokenizer(sentence)
        word_counter.update(tokens)

    if max_vocab_size is None:
        max_vocab_size = len(word_counter)

    if is_target:
        vocab['_GO'] = 0
        vocab['_PAD'] = 1
        vocab_idx = 2

        for key ,value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1
    else:
        vocab['_PAD'] = 0
        vocab_idx = 1
        for key,value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx +=1
    
    for key,value in vocab.items():
        reverse_vocab[value] = key
    
    return vocab,reverse_vocab,max_vocab_size

def token2idx(word, vocab):
    return vocab[word]

def sent2idx(sent, vocab, max_sentence_length, is_target=False):
    tokens = tokenizer(sent)
    current_length = len(tokens)
    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [1] * pad_length
    else:
        return [token2idx(token, vocab) for token in tokens] + [0] * pad_length, current_length

def idx2token(idx, reverse_vocab):
    return reverse_vocab[idx]

def idx2sent(indices, reverse_vocab):
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])

def show_loss(loss_history):
    plt.figure(figsize=(20, 10))
    plt.scatter(range(len(loss_history)), loss_history)
    plt.title('Learning Curve')
    plt.xlabel('Global step')
    plt.ylabel('Loss')
    plt.show()