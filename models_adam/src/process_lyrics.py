#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""process_lyrics.py
contains transformers dataset mapping functions to process lyrics
"""

__author__ = "Adam J. Weintraut"
__version__ = "0.0.1"

import pandas as pd
from datasets import Dataset
import re
import unicodedata

from src.nlp import *

'''
clean_lyrics():
        - hf mapping function to remove bad characters from lyrics
'''
def clean_lyrics(row):
    lyrics = row['lyrics']
    # remove lyric annotations, typically surrounded by
    lyrics = re.sub(r'\[{1,}.*?\]{1,}', '', lyrics)         # brackets
    lyrics = re.sub(r'\({1,}.*?\){1,}', '', lyrics)         # parentheses
    lyrics = re.sub(r'\*{1,}.*?\*{1,}', '', lyrics)         # asterisks
    # remove odd punctuation
    lyrics = lyrics.replace(u'\xa0', ' ')                   # &nbsp
    lyrics = lyrics.replace(u'\\xa0', ' ')                  # &nbsp
    lyrics = re.sub(r"[^a-zA-Z0-9 .,-[\n]]", r"", lyrics)  # characters except .,- \n
    lyrics = re.sub(r'\.{2,}', '.', lyrics)                 # .. ...
    # remove dashes & replace with space
    lyrics = lyrics.replace("_", "")
    # remove extra whitespace caused by above
    lyrics = re.sub(' +', ' ', lyrics)                      # repeating space chars
    lyrics = re.sub(r'\n+', '\n', lyrics).strip()           # repeating/extra newline chars
    lyrics = lyrics.replace('\n \n', '\n')                  # edge case
    lyrics = lyrics.replace('\n', ' \n ')                   # add in spacing
    # fix capitalization
    lyrics = lyrics.lower()
    # remove apostrophes that split during tokenization
    lyrics = lyrics.replace("'", "")

    # extra safe (trying to remove \xao char)
    lyrics = unicodedata.normalize("NFKD", lyrics)

    return {'clean_lyrics': lyrics}
    

'''
get_lyric_chunks():
            - hf mapping function to convert lyrics to chunks containing {n_lines} lines
'''
def get_lyric_chunks(row, n_lines):
    chunks = [' \n '.join(chunk) for chunk in list(split_into_chunks(get_lyric_lines(row['clean_lyrics']), n_lines))]
    return {'lyric_chunks': chunks}


'''
explode_chunks():
            - batched hf mapping function to create rows for each lyric chunk while keeping group columns constant
'''
def explode_chunks(batch, group):
  obj = {}
  # add constant groupby for each chunk
  for g in group:
    obj[g] = [_g for i, _g in enumerate(batch[g]) for _ in batch['lyric_chunks'][i]]
  # add lyric chunk + id
  obj["clean_lyrics"] = [chunk for chunk_list in batch["lyric_chunks"] for chunk in chunk_list]
  obj["lyric_chunk_n"] = [i for chunk_list in batch["lyric_chunks"] for i, chunk in enumerate(chunk_list)]

  return obj

'''
get_preceding_chunk():
    - get previous chunk (esp useful for 1 line chunks)
'''
def get_preceding_chunk(data):
    df = data.to_pandas()
    df['prev_clean_lyrics'] = df.sort_values(['midi_id', 'lyric_chunk_n']).groupby(['midi_id'])['clean_lyrics'].shift(1)
    df.fillna(value='', inplace=True)
    # reload so we can create inputs
    return Dataset.from_pandas(df)

'''
chunk_pipeline():
    - create chunks of n_lines using 'clean_lyrics'
'''
def chunk_pipeline(data, n_lines, group):
    # split lyrics into chunks (create 'lyric_chunks')
    data = data.map(get_lyric_chunks, fn_kwargs={'n_lines': n_lines})
    # explode chunks into their own rows (groupby / replace 'clean_lyrics' / remove 'lyric_chunks')
    data = data.map(explode_chunks, batched=True, fn_kwargs={'group': group}, remove_columns=['lyric_chunks'])
    # get preceding lyric chunk
    
    return data
