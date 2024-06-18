#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""iterator.py
contains Data Iterator class used to reduce GPU load via batched loading from disk memory
"""

__author__ = "Adam J. Weintraut"
__version__ = "0.0.1"

import numpy as np
import pandas as pd

class DataIterator:
    def __init__(self,
                 cfg,
                 tokenizer,
                 tensor_type,
                 n_examples,
                 max_load_at_once,
                 data_filename,
                 orig_target_cols=('orig', 'target'),
                 max_length=1028,
                 shuffle=True):
        # initialize
        self.cfg = cfg,
        self.tokenizer = tokenizer
        self.tensor_type = tensor_type
        self.n_examples = n_examples
        self.max_load_at_once = max_load_at_once
        self.data_filename = data_filename
        self.orig_target_cols = orig_target_cols
        self.max_length = max_length
        self.shuffle = shuffle

        # Initialize row order, call on_epoch_end to shuffle row indices
        self.row_order = np.arange(1, self.n_examples+1)
        self.on_epoch_end()

        # Load first chunk of max_load_at_once examples
        self.df_curr_loaded = self._load_next_chunk(0)
        self.curr_idx_in_load = 0

    def _load_next_chunk(self, idx):
        load_start = idx
        load_end = idx + self.max_load_at_once

        # Indices to skip are the ones in the shuffled row_order before and
        # after the chunk we'll use for this chunk
        load_idx_skip = self.row_order[:load_start] + self.row_order[load_end:]
        self.df_curr_loaded = pd.read_csv(self.data_filename, skiprows=load_idx_skip)
        self.df_curr_loaded = self.df_curr_loaded.sample(frac=1)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        if self.df_curr_loaded is None or self.curr_idx_in_load >= len(self.df_curr_loaded):
            self._load_next_chunk(idx)
            self.curr_idx_in_load = 0

        # orig target pair
        orig_target_pair = self.df_curr_loaded[list(self.orig_target_cols)].values.astype(str)[self.curr_idx_in_load]
        self.curr_idx_in_load += 1

        # call preprocessing
        item_data = self.preprocess_data(orig_target_pair)

        return item_data

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__()-1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.row_order = list(np.random.permutation(self.row_order))

    def preprocess_data(self, data):
      # extract orig/target
      orig, target = data
      # extra safe
      orig = orig.replace("'", "")
      orig = orig.replace("-", " ")

      # encode orig
      orig_encoded = self.tokenizer.batch_encode_plus(
          [orig],
          max_length=self.max_length,
          pad_to_max_length=True,
          padding='max_length',
          truncation=True,
          return_attention_mask=True,
          return_tensors=self.tensor_type,
      )

      # encode target
      target_encoded = self.tokenizer.batch_encode_plus(
          [target],
          max_length=self.max_length,
          pad_to_max_length=True,
          padding='max_length',
          truncation=True,
          return_attention_mask=True,
          return_tensors=self.tensor_type,
      )

      # extract input ids & attention masks
      orig_ids = orig_encoded['input_ids'].squeeze()
      orig_mask = orig_encoded['attention_mask'].squeeze()
      # orig_types = orig_encoded['token_type_ids'][0]
      target_ids = target_encoded['input_ids'].squeeze()
      target_mask = target_encoded['attention_mask'].squeeze()

      # remove effect of padding on loss function
      # target_ids = target_ids[:, :-1].contiguous()
      # target_ids[target_ids[:, 1:] == self.tokenizer.pad_token_id] = -100
      # labels[labels == self.tokenizer.pad_token_id] = -100

      return {'input_ids': orig_ids, 'attention_mask': orig_mask, 'labels': target_ids}