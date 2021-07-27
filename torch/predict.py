#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : predict.py
# Author            : Yan <yanwong@126.com>
# Date              : 16.07.2021
# Last Modified Date: 27.07.2021
# Last Modified By  : Yan <yanwong@126.com>

import argparse

import torch
from torchtext.data.utils import get_tokenizer

import datasets

parser = argparse.ArgumentParser(description='Predict by the saved model.')
parser.add_argument('model', help='path of the saved model')
parser.add_argument('vocab', help='path of the saved vocab')
parser.add_argument('device', choices=['cuda', 'cpu'], help='cuda or cpu')
parser.add_argument('input', help='text file, each line will be predicted by the model')
parser.add_argument('output', help='prediction result file')
args = parser.parse_args()

vocab = torch.load(args.vocab)
model = torch.load(args.model)
model.eval()  # important

tokenizer = get_tokenizer(datasets.chinese_tokenizer, language='chn')
text_transform = lambda x: [vocab[token] for token in tokenizer(x)]

softmax = torch.nn.Softmax(dim=1)

with open(args.input, 'r', encoding='utf-8') as (fin
    ), open(args.output, 'w', encoding='utf-8') as fout:
  for row in fin:
    row = torch.tensor(text_transform(row.strip())).to(args.device)
    row = torch.unsqueeze(row, 0)  # (1, seq_len)
    pred = model(row)  # (1, class_num)
    pred = softmax(pred)
    pred = pred.detach().to('cpu').numpy()
    fout.write('\t'.join(list(map(str, pred[0]))) + '\n')

