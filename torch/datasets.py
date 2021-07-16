#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : datasets.py
# Author            : Yan <yanwong@126.com>
# Date              : 09.07.2021
# Last Modified Date: 14.07.2021
# Last Modified By  : Yan <yanwong@126.com>

from collections import Counter, OrderedDict

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab, vocab

class CtripSentimentDataset(Dataset):
  def __init__(self, annotations_file):
    self.annotations = pd.read_csv(annotations_file, header=0)

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, idx):
    label = self.annotations.iloc[idx, 0]
    text = self.annotations.iloc[idx, 1]
    return text, label

def chinese_tokenizer(text):
  return [tok for tok in text]

def build_vocab(dataset, tokenizer):
  counter = Counter()
  size = len(dataset)
  for i in range(size):
    text, label = dataset[i]
    counter.update(tokenizer(text))

  sorted_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
  ordered_dict = OrderedDict(sorted_tuples)
  v = vocab(ordered_dict)
  pad_token = '<PAD>'
  unk_token = '<UNK>'
  v.insert_token(pad_token, 0)
  v.insert_token(unk_token, 1)
  v.set_default_index(v[unk_token])
  return v

