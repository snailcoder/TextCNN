import numpy as np
import re
from collections import defaultdict

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(pos_path, neg_path):
    sents = []
    labels = []
    vocab = defaultdict(int)
    with open(pos_path, "r") as (pos_file
      ), open(neg_path, "r") as (neg_file
      ):
        for l in pos_file:
            s = clean_str(l.strip())
            words = set(s.split())
            for w in words:
                vocab[w] += 1
            sents.append(s)
            # tf.nn.softmax_cross_entropy_with_logits takes labels
            # in the form of one-hot vectors.
            labels.append((1, 0))
        for l in neg_file:
            s = clean_str(l.strip())
            words = set(s.split())
            for w in words:
                vocab[w] += 1
            sents.append(s)
            labels.append((0, 1))
    return list(zip(sents, labels)), vocab

def get_word_idx_map(vocab):
    word_idx_map = {"<PAD>": 0}
    i = 1
    for w in vocab:
        word_idx_map[w] = i
        i += 1
    return word_idx_map

def fetch_batch(data, batch_index, batch_size):
    data_size = len(data)
    # num_batches = int(np.ceil(data_size / batch_size))
    beg = batch_index * batch_size
    end = min((batch_index + 1) * batch_size, data_size)
    return data[beg : end]
    