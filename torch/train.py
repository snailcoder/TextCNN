#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train.py
# Author            : Yan <yanwong@126.com>
# Date              : 09.07.2021
# Last Modified Date: 16.07.2021
# Last Modified By  : Yan <yanwong@126.com>

import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

import model
import datasets
import config

parser = argparse.ArgumentParser(
    description='Train a TextCnn model and save the best weights.')
parser.add_argument('data_file', help='path of dataset')
parser.add_argument('data_name', choices=['ctrip'], help='name of dataset')
parser.add_argument('save_dir', help='directory tor save the best model')
parser.add_argument('--log_interval', type=int, default=100,
                    help='print training log every interval batches')
args = parser.parse_args()

dataset = None
if args.data_name == 'ctrip':
  dataset = datasets.CtripSentimentDataset(args.data_file)

if dataset is None:
  raise ValueError(f'Invalid dataset type: {args.data_name}')

train_config = config.TrainConfig()

train_size = int(train_config.train_size * len(dataset))
test_size = len(dataset) - train_size
training_data, test_data= torch.utils.data.random_split(dataset, [train_size, test_size])

tokenizer = get_tokenizer(datasets.chinese_tokenizer, language='chn')
vocab = datasets.build_vocab(training_data, tokenizer)

text_transform = lambda x: [vocab[token] for token in tokenizer(x)]
label_transform = lambda x: int(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

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

model_config = config.ModelConfig()
model = model.TextCnn(model_config).to(device)
 
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
         ' Avg loss: {test_loss:>8f} \n')
  return accuracy

def save_model(model, save_dir):
  if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, 'best_weights.pth')
  torch.save(model.state_dict(), save_path)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=train_config.learning_rate)

best_accu = 0
for t in range(train_config.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    accu = test_loop(test_dataloader, model, loss_fn)
    if accu > best_accu:
      best_accu = accu
      save_model(model, args.save_dir)
      print(f'Best accuracy: {best_accu}\n')
print("Done!")

