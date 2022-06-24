#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train.py
# Author            : Yan <yanwong@126.com>
# Date              : 09.07.2021
# Last Modified Date: 16.06.2022
# Last Modified By  : Yan <yanwong@126.com>

import os
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

import model
import datasets
import config

parser = argparse.ArgumentParser(
    description='Train a TextCnn model and save the best weights.')
parser.add_argument('data_file', help='dataset file')
parser.add_argument('data_name', choices=['ctrip'], help='dataset name')
parser.add_argument('save_dir', help='directory to save vocab and model')
parser.add_argument('--word_vec', help='pretrained word vector file')
parser.add_argument('--log_interval', type=int, default=100,
                    help='print training log every interval batches')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

dataset = None
if args.data_name == 'ctrip':
  dataset = datasets.CtripSentimentDataset(args.data_file)

if dataset is None:
  raise ValueError(f'Invalid dataset type: {args.data_name}')

model_config = config.ModelConfig()
train_config = config.TrainConfig()

train_size = int(train_config.train_size * len(dataset))
test_size = len(dataset) - train_size
training_data, test_data= torch.utils.data.random_split(dataset, [train_size, test_size])

tokenizer = get_tokenizer(datasets.chinese_tokenizer, language='chn')
vocab = datasets.build_vocab(training_data, tokenizer)

def save_model(model, save_dir, filename):
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, filename)
  # torch.save(model.state_dict(), save_path)
  torch.save(model, save_path)

save_model(vocab, args.save_dir, 'vocab.pth')

text_transform = lambda x: [vocab[token] for token in tokenizer(x)]
label_transform = lambda x: int(x)

def collate_batch(batch):
  label_list, text_list = [], []
  for (_text, _label) in batch:
    processed_text = torch.tensor(text_transform(_text))
    text_list.append(processed_text)
    label_list.append(label_transform(_label))
  label_list = torch.tensor(label_list)
  return pad_sequence(text_list, batch_first=True).to(device), label_list.to(device)

train_dataloader = DataLoader(training_data,
                              batch_size=train_config.batch_size,
                              shuffle=train_config.shuffle,
                              collate_fn=collate_batch)
test_dataloader = DataLoader(test_data,
                             batch_size=train_config.batch_size,
                             shuffle=train_config.shuffle,
                             collate_fn=collate_batch)

def load_pretrained_embedding(filename, emb_dim, words):
  w2v = {}
  ws = set(words)
  with open(filename, 'r', encoding='utf-8') as f:
    for row in f:
      toks = row.split(' ')
      if toks[0] in ws:
        w2v[toks[0]] = np.array(list(map(float, toks[-emb_dim:])))
  emb = []
  oop_num = 0
  for w in words:
    if w not in w2v:
      w2v[w] = np.random.uniform(-0.25, 0.25, emb_dim)
      oop_num += 1
    emb.append(w2v[w])

  print(f'# out-of-pretrained words: {oop_num}')

  return emb

pretrained_embedding = None
if args.word_vec:
  pretrained_embedding = load_pretrained_embedding(
      args.word_vec, model_config.d_word, vocab.get_itos())
  print('# total words: %d' % len(pretrained_embedding))
  pretrained_embedding = torch.FloatTensor(pretrained_embedding).to(device)

model = model.TextCnn(model_config, len(vocab), pretrained_embedding).to(device)
 
def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    if batch > 0 and batch % args.log_interval == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  accuracy = correct / size
  print(f'Test Error: \n Accuracy: {(100*accuracy):>0.1f}%,'
        f' Avg loss: {test_loss:>8f} \n')
  return accuracy


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=train_config.learning_rate)

best_accu = 0
for t in range(train_config.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    accu = test_loop(test_dataloader, model, loss_fn)
    if accu > best_accu:
      best_accu = accu
      save_model(model, args.save_dir, 'best_model.pth')
      print(f'Best accuracy: {best_accu}\n')
print(f'Global best accuracy: {best_accu}')

