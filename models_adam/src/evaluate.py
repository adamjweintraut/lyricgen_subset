#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""evaluate.py
contains functions generate/evaluate model results
"""

__author__ = "Adam J. Weintraut"
__version__ = "0.0.1"

from datasets import load_metric
import rouge_score
from sentence_transformers import SentenceTransformer
import torch

from src.hf import save_eval_data, load_hf_dataset
from src.gpu import clear_gpu
from src.nlp import *


'''
print_eval_summary():
    - takes in hf dataset and prints out examples w/ summary statistics
'''
def print_eval_summary(eval):
  for i, row in enumerate(eval):
    print(f'''\
    {100*round(row['syl_error'], 2)}
    - pred syls: {row['predicted_syls']}
    - label syls: {row['label_syls']}
    ------------------------------------
    - input: {row['orig']}
    ------------------------------------
    - pred: {row['predicted']}
    - label: {row['target']}
    ''')



'''
generate_kwsyl_lyrics():
        - hf mapping function to generate lyric plan
'''
def generate_kwsyl_lyrics(row, model, tokenizer, device, cfg, genconfig):
    args = cfg['args']
    params = cfg['params']
    input = params['orig_target_cols'][0]

    # print('rowinput', row[input])
    # print('check target',row[params['orig_target_cols'][1]])
    
    # encode
    encodings = tokenizer(row[input],
                          truncation=args['truncation'],
                          padding=args['padding'],
                          return_tensors=params['tensor_type'])

    # for item in encodings['input_ids'].flatten().tolist():
    #   print('token ',item,':',tokenizer.decode([item]))

    # print('attention_mask', encodings["attention_mask"].flatten().tolist())

    # generate
    with torch.no_grad():
      generated = model.generate(input_ids=encodings["input_ids"].to(device),
                                 attention_mask=encodings["attention_mask"].to(device),
                                 generation_config=genconfig)

      # for item in generated.flatten().tolist():
      #   print('token:',item,'decoded:',tokenizer.decode(item,skip_special_tokens=False))


      decoded = tokenizer.batch_decode(generated,
                                       skip_special_tokens=args['skip_special_tokens'],
                                       clean_up_tokenization_spaces=args['clean_up_tokenization_spaces']
      )
    return {'predicted': decoded}


'''
extract_nested_string_fields()
      - helper to extract nested fields for metrics
'''
def extract_nested_string_fields(row):
  for col in row.keys(): row[col] = row[col][0] if type(row[col]) is list else row[col]
  return row


'''
load_metrics_models()
      - load models we will use for computing metrics
'''
def load_metrics_models():
   # load evaluation models
  rouge = load_metric("rouge")
  sent_sim = SentenceTransformer('distilbert-base-uncased-finetuned-sst-2-english')

  return rouge, sent_sim


'''
compute_rouge()
    - https://stats.stackexchange.com/questions/301626/interpreting-rouge-scores
'''
def compute_rouge(row, rouge, quantiles, metrics, orig_target_cols):
  results = rouge.compute(predictions=[row['predicted']], references=[row[orig_target_cols[1]]], use_stemmer=True)['rougeL']
  for q in range(len(quantiles)):
    for m in range(len(metrics)):
      row[f'rougeL_{quantiles[q]}_{metrics[m]}'] = results[q][m]
  return row


'''
compute_predicted_label_similarity()
    - https://medium.com/@tanner.overcash/semantic-similarity-calculations-using-nlp-and-python-a-soft-introduction-1f31df965e40
'''
def compute_predicted_label_similarity(row, sent_sim, orig_target_cols):
  emb = sent_sim.encode([row['predicted'], row[orig_target_cols[1]]])
  return {f"predicted_{orig_target_cols[1]}_sim": emb[0].dot(emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))}


'''
compute_syllable_metrics()
      - calculate sylls/line & perform metrics
'''
def compute_syllable_metrics(ex):
  return {'predicted_syls': count_line_syllables(ex['predicted']),
          'label_syls': count_line_syllables(ex['target']),
          'syl_ape': np.abs(ex['predicted_syls'] - ex['label_syls']) / ex['label_syls']
  }

'''
compute_kwsylgen_metrics():
  - computes evaluation metrics using HF Dataset map api
'''
def compute_kwsylgen_metrics(data, cfg):
  # load metrics models
  rouge, sent_sim = load_metrics_models()
  # cleanup
  data = data.map(extract_nested_string_fields)
  ## evaluation metrics
  ######################
  # rouge
  data = data.map(compute_rouge, fn_kwargs={'rouge':rouge, 
                                            'quantiles':['median','max'], 
                                            'metrics':['precision','recall','fmeasure'],
                                            'orig_target_cols' : cfg['params']['orig_target_cols']})
  # sentence similarity
  data = data.map(compute_predicted_label_similarity, fn_kwargs={'sent_sim': sent_sim, 
                                                                 'orig_target_cols': cfg['params']['orig_target_cols']})
  # syllable count error
  data = data.map(compute_syllable_metrics)
  # cleanup
  data = data.map(extract_nested_string_fields)

  return data


'''
generate_and_evaluate()
      - runner
'''
def generate_and_evaluate(cfg, n_examples=None, write_data=False, metrics_msg="without metrics"):
  # load our test dataset
  data = load_hf_dataset(hf_path = cfg['model']['hf_dataset'], dotenv = cfg['dotenv'], 
                         split = 'test', 
                         n_examples = n_examples)
  # load hf model objs
  model, tokenizer, device, genconfig = load_hf_model_objects(cfg['model']['hf_full_path'], cfg)
  # generate predictions
  data = data.map(generate_kwsyl_lyrics, fn_kwargs={'model': model,
                                                    'tokenizer': tokenizer,
                                                    'device': device,
                                                    'cfg': cfg,
                                                    'genconfig': genconfig})
  # compute eval metrics
  try:
    data = compute_kwsylgen_metrics(data, cfg)
    metrics_msg = "with metrics"
  except Exception as e:
    print(f"Unexpected {e=}, {type(e)=}")
    print("--------------------------------------")
    print("you will need to compute metrics again")
    print("--------------------------------------")
  
  if write_data:
      # save data
      save_eval_data(data=data,
                     dotenv=cfg['dotenv'],
                     hf_write_path=cfg['model']['hf_eval_dataset'],
                     local_write_path=cfg['model']['eval_filepath'],
                     message=f"generated {n_examples} predictions {metrics_msg} | {str(cfg['args'])}")
  # clear space
  clear_gpu()

  return data