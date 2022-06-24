#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : model.py
# Author            : Yan <yanwong@126.com>
# Date              : 07.07.2021
# Last Modified Date: 24.06.2022
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
      nn.Conv2d(1, c, (h, config.d_word), padding=0)
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
    # There's no problem with the following code if you just want to train a
    # pytorch model that can work properly:
    #
    # x = [F.max_pool1d(c, c.size()[2]).squeeze(2) for c in x]
    #
    # However, when export this model to ONNX, it throws an exception:
    #
    # TypeError: max_pool1d_with_indices(): argument 'kernel_size' (position 2)
    # must be tuple of ints, not Tensor
    #
    # You can fix this error by casting c.size()[2] to integer like this issue
    # https://github.com/pytorch/pytorch/issues/11296 said:
    #
    # x = [F.max_pool1d(c, int(c.size()[2])).squeeze(2) for c in x]
    #
    # Unfortunately, this leads to a new problem: when export to ONNX, this 
    # argument will be converted to a constant, which will lead to wrong result
    # because the input sequence length for inference may be different from that
    # for tracing. See https://pytorch.org/docs/stable/onnx.html#avoiding-pitfalls .
    # So I use adaptive_max_pool1d instead of max_pool1d.
    x = [F.adaptive_max_pool1d(c, 1).squeeze(2) for c in x]  # [(N, C_out), ...]
    x = torch.cat(x, dim=1)  # (N, sum(C_out))
    x = self.dropout(x)  # (N, sum(C_out))
    logits = self.ff(x)
    return logits

