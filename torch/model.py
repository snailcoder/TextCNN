#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : model.py
# Author            : Yan <yanwong@126.com>
# Date              : 07.07.2021
# Last Modified Date: 18.07.2021
# Last Modified By  : Yan <yanwong@126.com>

import torch
from torch import nn
import torch.nn.functional as F

class TextCnn(nn.Module):
  def __init__(self, config, vocab_size, pretrained_embedding):
    super(TextCnn, self).__init__()
    if pretrained_embedding is not None:
      self.embedding = nn.Embedding.from_pretrained(
          pretrained_embedding, freeze=config.static_embedding)
    else:
      self.embedding = nn.Embedding(vocab_size, config.d_word)
    self.convs = nn.ModuleList([
      nn.Conv2d(1, c, (h, config.d_word), padding='valid')
                for c, h in zip(config.filter_num, config.filter_heights)])
    self.dropout = nn.Dropout(config.dropout)
    self.ff = nn.Linear(sum(config.filter_num), config.class_num)

  def forward(self, x):
    # N: batch size
    # H_in: sequence length
    # W_in: dimension of word embedding
    # H_out: H_in - filter_size + 1
    # C_out: number of filter

    # x.shape == (N, H_in)
    x = self.embedding(x)  # (N, H_in, W_in)
    x = x.unsqueeze(1)  # (N, 1, H_in, W_in)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, C_out, H_out), ...]
    x = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in x]  # [(N, C_out), ...]
    x = torch.cat(x, dim=1)  # (N, sum(C_out))
    x = self.dropout(x)  # (N, sum(C_out))
    logits = self.ff(x)
    return logits

