#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""hf.py
contains functions to load/save huggingface datasets/model objects
"""

__author__ = "Adam J. Weintraut"
__version__ = "0.0.1"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
import gc
from huggingface_hub import login

'''
load_csv_dataset():
        - load csv file as transformers dataset
'''
def load_csv_dataset(filename, n_examples):
    split = 'train' if n_examples is None else f'train[:{n_examples}]'
    data = load_dataset('csv', data_files=filename, split=split)

    return data


'''
save_csv_dataset():
        - save transformers dataset as csv file
'''
def save_csv_dataset(data, local_write_path):
    # write to csv
    try:
        data.to_csv(local_write_path)
    except Exception as e:
        print('save to csv failed')
        print(e)
        return False

    return True
    
    
'''
load_hf_dataset():
        - load hf dataset as transformers dataset
'''
def load_hf_dataset(hf_path, dotenv, split='train', n_examples=None, shuffle=False):
    if shuffle:
        data = load_dataset(hf_path, split=split, token=dotenv['hf_read_token_adam']).flatten()
        data = data.shuffle(seed=42)
        data = data.select(range(int(n_examples))) if n_examples else data
    else:
        split = f'{split}[:{n_examples}]' if n_examples else f'{split}[:{n_examples}]'
        data = load_dataset(hf_path, split=split, token=dotenv['hf_read_token_adam']).flatten()

    return data


'''
save_hf_dataset():
        - save transformers dataset in hf
'''
def save_hf_dataset(data, dotenv, hf_write_path, message="Uploading"):
    # write to hf
    try:
        data.push_to_hub(hf_write_path,
                         token=dotenv['hf_write_token_adam'],
                         commit_message=message)
    except Exception as e:
        print('save to hf failed')
        print(e)
        return False

    return True
    
    
'''
add_tokens():
      - add special tokens to tokenizer
'''
def add_tokens(model, tokenizer, cfg):
  # set pad token to be eos token
  tokenizer.pad_token = tokenizer.eos_token
  # add tokens
  tokens = ["<P>"]
  if cfg['params']['add_syllable_tokens']:
    tokens += [f'len_{i}' for i in np.arange(1,30)]
  # add to tokenizer & resize model tokens to match
  # tokenizer.add_special_tokens({'additional_special_tokens': tokens})
  tokenizer.add_tokens(tokens)
  model.resize_token_embeddings(len(tokenizer))

  return model, tokenizer


'''
load_hf_model_objects():
        - return model, tokenizer, and device
'''
def load_hf_model_objects(hf_model, cfg):
    # instaniate model & update generation configs
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_model, token=cfg['dotenv']['hf_read_token_adam'])
    genconfig = GenerationConfig(**cfg['args'])
    model.generation_config = genconfig
    # load tokenizer and modify tokens
    tokenizer = AutoTokenizer.from_pretrained(hf_model, token=cfg['dotenv']['hf_read_token_adam'])
    # modify token library
    # model, tokenizer = add_tokens(model, tokenizer, cfg)
    # save model & tokenizer to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, tokenizer, device, genconfig


'''
push_model_to_huggingface():
        - push trainer, tokenizer, genconfig to hf
'''
def push_model_to_huggingface(cfg, trainer, tokenizer, genconfig=None):
  try:
    login(cfg['dotenv']['hf_write_token_adam'])
    trainer.push_to_hub(cfg['model']['hf_path'])
    tokenizer.push_to_hub(cfg['model']['hf_path'])
    if genconfig:
      genconfig.save_pretrained(cfg['model']['hf_path'], push_to_hub=True)
    print('chaaaaaa ching!')
  except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")
    print("-----------------------------------")
    print('you will need to rewrite this model')
    print("-----------------------------------")


'''
save_eval_data():
'''
def save_eval_data(data, dotenv, hf_write_path, local_write_path, message="Uploading"):
    # attempt to save
    hf_write_success = save_hf_dataset(data, dotenv, hf_write_path, message=message)
    csv_write_success = save_csv_dataset(data, local_write_path)
    # cumulative success flag
    save_successful = (hf_write_success and csv_write_success)
    # echo confirmation
    print('great success!') if save_successful else print('aawwwww maaannnnn! did ya forget to set write_data?')