#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""train.py
contains functions to orchestrate hf/transformers training pipeline
"""

__author__ = "Adam J. Weintraut"
__version__ = "0.0.1"

from ast import Return
import evaluate
import accelerate
import numpy as np
import pandas as pd
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, TrainingArguments  #, default_data_collator 

from src.hf import load_hf_model_objects, push_model_to_huggingface
from src.iterator import DataIterator
from src.nlp import postprocess_text
from src.gpu import clear_gpu


'''
compute_training_metrics():
      - show metrics during training stages
'''
def compute_training_metrics(tokenizer, rouge):
  def compute_metrics(eval_preds):
      preds, labels = eval_preds
      if isinstance(preds, tuple):
          preds = preds[0]
      # Replace -100s used for padding as we can't decode them
      preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
      labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

      # Some simple post-processing
      # decoded_preds, decoded_labels = postprocess_text(tokenizer.batch_decode(preds, skip_special_tokens=True),
      #                                                  tokenizer.batch_decode(labels, skip_special_tokens=True))
      # compute metrics
      result = rouge.compute(predictions=preds,
                              references=labels,
                              tokenizer=tokenizer)
      # round results
      result = {k: round(v, 4) for k, v in result.items()}
      # add in generated length
      prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
      result["gen_len"] = np.mean(prediction_lens)
      return result

  return compute_metrics


'''
get_data_iterators():
      - insantiate data iterators for train/valid sets
'''
def get_data_iterators(cfg, model, tokenizer):
  # training data
  train_data_iterator = DataIterator(
                            cfg = cfg,
                            tokenizer = tokenizer,
                            tensor_type = cfg['params']['tensor_type'],
                            max_load_at_once = cfg['params']['max_load_at_once'],
                            data_filename = cfg['params']['train_data_filepath'],
                            orig_target_cols = cfg['params']['orig_target_cols'],
                            n_examples = pd.read_csv(cfg['params']['train_data_filepath']).shape[0],
                            max_length = cfg['params']['max_length'],
                         )
  # validation data
  valid_data_iterator = DataIterator(
                            cfg = cfg,
                            tokenizer = tokenizer,
                            tensor_type = cfg['params']['tensor_type'],
                            max_load_at_once = cfg['params']['max_load_at_once'],
                            data_filename = cfg['params']['valid_data_filepath'],
                            orig_target_cols = cfg['params']['orig_target_cols'],
                            n_examples = pd.read_csv(cfg['params']['valid_data_filepath']).shape[0],
                            max_length = cfg['params']['max_length'],
                         )

  return (train_data_iterator, valid_data_iterator)


'''
train_pytorch_model():
      - runner
'''
def train_pytorch_model(cfg, model, tokenizer, genconfig, train_data_iterator, valid_data_iterator):
  # create rouge eval
  rouge = evaluate.load("rouge")
  # create trainer object
  trainer = Seq2SeqTrainer(
                model = model,
                train_dataset = train_data_iterator,
                eval_dataset = valid_data_iterator,
                # compute_metrics = compute_training_metrics(tokenizer, rouge),
                args = Seq2SeqTrainingArguments(
                            push_to_hub = True,
                            push_to_hub_model_id = cfg['model']['hf_path'],
                            push_to_hub_token = cfg['dotenv']['hf_write_token_adam'],
                            hub_strategy = 'checkpoint',
                            output_dir = cfg['model']['filepath'],
                            overwrite_output_dir = True,
                            evaluation_strategy = 'steps',
                            eval_steps = 500,
                            save_steps = 500,
                            logging_steps = 500,
                            save_total_limit = 3,
                            load_best_model_at_end = True,
                            learning_rate = 2e-5,
                            per_device_train_batch_size = cfg['params']['batch_size'],
                            per_device_eval_batch_size = cfg['params']['batch_size'],
                            num_train_epochs = cfg['params']['num_train_epochs'],
                            predict_with_generate = True,
                            fp16 = cfg['params']['fp16'],
                            report_to="wandb",
                        ),
              )
  try:
    # train model & save to hf
    resume = cfg['model']['last_checkpoint'] if cfg['resume_from_checkpoint'] else False
    trainer.train(resume_from_checkpoint=resume)
    push_model_to_huggingface(cfg, trainer, tokenizer, genconfig)
  except Exception as e:
    print(e)

  return trainer, tokenizer

'''
run_training_pipeline():
        - runner
'''
def run_training_pipeline(cfg):
  print(cfg)
  # load pretrained model
  model, tokenizer, device, genconfig = load_hf_model_objects(cfg['model']['pretrain'], cfg)
  # load data iterators for pytorch
  train_data_iterator, valid_data_iterator = get_data_iterators(cfg, model, tokenizer)
  # train pytorch model
  try:
    trainer, tokenizer = train_pytorch_model(cfg, model, tokenizer, genconfig, train_data_iterator, valid_data_iterator)
  except Exception as e:
      print(f'Exception: {e}')
      model, tokenizer, device, genconfig = None, None, None, None
      clear_gpu()
  return model, trainer, tokenizer, genconfig