#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""config.py
contains functions configure model training/generation/eval
"""

__author__ = "Adam J. Weintraut"
__version__ = "0.0.1"

from datetime import datetime
from dotenv import dotenv_values
import os 

'''
configure_batches():
      - set batch size based on max length
'''
def configure_batch_size(max_length):
  # return max_length, (32 / (max_length / 64))
  match max_length:
    case 64:
      return (64, 64)
    case 128:
      return (128, 16)
    case 256:
      return (256, 8)
    case 512:
      return (64, 4)
    case _:
      return (-1,-1)


'''
get_orig_target_cols():
    - get cols used for orig/target dynamically 
'''
def get_orig_target_cols(dataset, variant):
  if dataset == 'lyrlen':
    return ('orig', 'target')
  elif dataset == 'loaf':
    if variant is None:
      return ('orig', 'target')
    elif variant == 'lyrictoplan':
      return ('plan_orig', 'plan_target')
  else:
    return ('orig', 'target')


'''
get_model_configs():
    - dynamically choose data source, model, and hyperparams. 
'''
def get_model_configs(user='adamjweintraut', model_type='bart', pretrain='facebook/bart-large', dataset='lyrlen', variant=None, max_length=256, epochs=2, tensor_type='pt', dotenv=None, resume_from_checkpoint=False):
    dotenv = dotenv_values(f'{ROOT}/.env') if not dotenv else dotenv
    # print(dotenv.keys())
    # find latest checkpoint
    cfg = {
            'resume_from_checkpoint': resume_from_checkpoint,
            'model': {
                          'hf_path': f"{model_type}-finetuned-{dataset}-{max_length}",
                          'hf_dataset' : f'{user}/{dataset}',
                          'pretrain': pretrain,
                          'type': model_type,
                          'filepath': f"{dotenv['models']}/{dataset}/{model_type}/{model_type}-finetuned-{dataset}-{max_length}",
            },
            'params': {
                          'train_data_filepath': dotenv[f'{dataset}_train'],
                          'valid_data_filepath': dotenv[f'{dataset}_valid'],
                          'orig_target_cols': get_orig_target_cols(dataset, variant),
                          'tensor_type': tensor_type,
                          'dataset' : dataset,
                          'max_length': max_length,
                          'max_load_at_once': configure_batch_size(max_length)[0],
                          'batch_size': configure_batch_size(max_length)[1],
                          'num_train_epochs': epochs,
                          'add_syllable_tokens': False if variant == 'lyrictoplan' else True,
                          'fp16': True,

            },
            'args': {
                          'n_examples': None,
                          'truncation': True,
                          'pad_to_max_length': True,
                          'padding': 'max_length',
                          "bos_token_id": 0,
                          "forced_bos_token_id": 0,
                          "decoder_start_token_id": 2,
                          "eos_token_id": 2,
                          "forced_eos_token_id": 2,
                          "pad_token_id": 2,
                          # 'min_new_tokens': max_length-80,
                          'max_new_tokens': max_length,
                          'early_stopping': True,
                          'num_beams': 4,
                          'do_sample': True,
                          'top_k': 0,
                          'top_p': 0.9,
                          'temperature': 0.85,
                          'no_repeat_ngram_size': 2,
                          'num_return_sequences': 1,
                          'repetition_penalty': 1,
                          'renormalize_logits': True,
                          'skip_special_tokens': True,
                          'clean_up_tokenization_spaces': True
            },
            'dotenv': dotenv,
    }
    # add variants if necessary
    if variant is not None:
      cfg['model']['hf_path'] += f'-{variant}'
      cfg['model']['filepath'] += f'-{variant}'
    # add metadata to configs that require a defined model path
    cfg['model']['hf_full_path'] = f"{user}/{cfg['model']['hf_path']}"
    cfg['model']['last_checkpoint'] = f"{cfg['model']['hf_full_path']}/last-checkpoint"
    cfg['model']['hf_eval_dataset'] = cfg['model']['hf_path'] + f"_{datetime.today().strftime('%Y-%m-%d')}_run"
    cfg['model']['eval_filepath'] = cfg['model']['filepath'] + f"_{datetime.today().strftime('%Y-%m-%d')}_run.csv"

    return cfg