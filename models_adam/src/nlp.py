#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""nlp.py
contains utility functions to process/format lyric text for modeling
"""

__author__ = "Adam J. Weintraut"
__version__ = "0.0.1"

import ast
from keybert import KeyBERT
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
import re
import yake

# cmu pronunciation dict (are these redundant?)
nltk.download('cmudict')
nltk.download('punkt')

# instantiate dictionary for phonetic syllables
syllable_dict = cmudict.dict()

# manual syllable counting
##################################################
'''
VOWEL_RUNS
    - count syllables for:
        - repeated vowels: sour
'''
VOWEL_RUNS = re.compile("[aeiouy]+", flags=re.I)

'''
EXCEPTIONS
    - count syllables for:
        - trailing e: smite, scared
        - adverbs with e: nicely
'''
EXCEPTIONS = re.compile("[^aeiou]e[sd]?$|" + "[^e]ely$", flags=re.I)

'''
ADDITIONAL
    - fixes errors from EXCEPTIONS like:
        - incorrect subtractions: smile, scarred, raises
        - misc issues: flying, piano, video, prism
'''
ADDITIONAL = re.compile("[^aeioulr][lr]e[sd]?$|[csgz]es$|[td]ed$|" + ".y[aeiou]|ia(?!n$)|eo|ism$|[^aeiou]ire$|[^gq]ua", flags=re.I)


'''
postprocess_text():
      - helper function for rouge calculation
'''
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


'''
count_word_syllables_man():
    - manually count syllables using ruleset
'''
def count_word_syllables_man(word):
    vowel_runs = len(VOWEL_RUNS.findall(word))
    exceptions = len(EXCEPTIONS.findall(word))
    additional = len(ADDITIONAL.findall(word))
    return max(1, vowel_runs - exceptions + additional)

'''
count_word_syllables_cmu():
    - uses cmu pronunciation dictionary
'''
def count_word_syllables_cmu(word):
  return [len(list(y for y in x if y[-1].isdigit())) for x in syllable_dict[word.lower()]][0]

'''
count_word_syllables():
'''
def count_word_syllables(word):
    try:
        return count_word_syllables_cmu(word)
    except KeyError:
        return count_word_syllables_man(word)

'''
count_line_syllables():
'''
def count_line_syllables(line):
    tokens = word_tokenize(line)
    syllable_counts = [count_word_syllables(token) for token in tokens if token.isalpha()]
    return sum(syllable_counts)
    

'''
uppercase():
        - regex matching function
'''
def uppercase(matchobj):
    return matchobj.group(0).upper()


'''
capitalize():
'''
def capitalize(s):
    return re.sub('^([a-z])|[\.|\?|\!]\s*([a-z])|\s+([a-z])(?=\.)', uppercase, s)


'''
fix_capitalization():
'''
def fix_capitalization(s):
    if s.isupper():
        s = s.lower()
    return capitalize(s)


'''
remove_punctuation():
'''
def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))


'''
get_lyric_lines():
        - split on newline and ignore empty lines
'''
def get_lyric_lines(lyrics):
    return [line.strip() for line in lyrics.split('\n') if line != '']


'''
split_into_chunks():
        - generator helper function to split a list into chunks
'''
def split_into_chunks(list, chunk_size):
    for i in range(0, len(list), chunk_size):
        yield list[i:i + chunk_size]

'''
get_kw_extractor():
        - load keyword extractor model
'''
def get_kw_extractor(top=3):
    return yake.KeywordExtractor(n=1, top=top)
    