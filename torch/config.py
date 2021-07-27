#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : config.py
# Author            : Yan <yanwong@126.com>
# Date              : 08.07.2021
# Last Modified Date: 19.07.2021
# Last Modified By  : Yan <yanwong@126.com>

class ModelConfig(object):
  def __init__(self):
    self.d_word = 300
    self.static_embedding = False
    self.filter_heights = [3, 4, 5]
    self.filter_num = [200, 200, 200]
    self.dropout = 0.5
    self.class_num = 2

class TrainConfig(object):
  def __init__(self):
    self.learning_rate = 1e-3
    self.batch_size = 32
    self.epochs = 10
    self.shuffle = True
    self.train_size = 0.9

