# models_adam

This directory contains the modules, notebooks, and directory structure used to train some of the models we experimented with. Most of these are BART models trained on Colab using HuggingFace. Models and Data are stored [here](https://huggingface.co/adamjweintraut)

- [models\_adam](#models_adam)
  - [preprocessing.ipynb](#preprocessingipynb)
  - [experiment\_pipeline.ipynb](#experiment_pipelineipynb)
  - [data](#data)
  - [notebooks](#notebooks)
  - [src](#src)
    - [config.py](#configpy)
    - [evaluate.py](#evaluatepy)
    - [gpu.py](#gpupy)
    - [hf.py](#hfpy)
    - [iterator.py](#iteratorpy)
    - [nlp.py](#nlppy)
    - [process\_lyrics.py](#process_lyricspy)
    - [train.py](#trainpy)


## preprocessing.ipynb
> Notebook used to preprocess / save data sources to huggingface. Uses modules from `src`, specifically `hf.py`, `process_lyrics.py`, and `nlp.py`. Currently references both local data files (from earlier preprocessing work) and huggingface datasets. 
> Requires a valid `.env` file, which contains tokens/secrets for hf and elasticsearch and maps data source labels to their relevant filepaths.

## experiment_pipeline.ipynb
> Notebook used to train models. Uses modules from `src`, namely `hf.py`, `config.py`, `iterator.py`, `train.py`, and `evaluate.py`. Dynamic, enabling training varying models with different hyperparams and data sources using `get_model_configs` from `config.py`.
> Requires a valid `.env` file, which contains tokens/secrets for hf and elasticsearch and maps data source labels to their relevant filepaths.

## data
> This holds csv files containing the train, test, and valid splits for each data source used to train the models, named as `{dataset}_{split}.csv`. This is important since the data iterator in `iterator.py` used in  `experiment_pipeline.ipynb` notebook references them dynamically.

## notebooks
> Contains some additional notebooks used for exploration (such as snippets for noising techniques & processing xml files) as well as earlier variants to `experiment_pipeline.py` that were model-specific. 

## src
> Contains modules used in `preprocessing.ipynb` and `experiment_pipeline.ipynb`. Their functions are as described below: 

### config.py
> Enables dynamic data sources, model variations, and hyperparameter tuning for modules by creating a `cfg` dictionary object.

### evaluate.py
> Contains functions necessary for generating and evaluating results.

### gpu.py
> Utility functions that print GPU utilization & clear out memory.

### hf.py
> Contains functions for loading/saving huggingface models & datasets. 

### iterator.py
> Contains the DataIterator class that allows us to reduce GPU usage during training by loading in batches from disk memory.

### nlp.py
> Contains a variety of helper/utility functions to process text, count syllables, and load relevant text corpuses.

### process_lyrics.py
> Contains functions used primarily by `preprocessing.ipynb` to clean lyrics from varying data sources, split lyrics into chunks of n lyric lines, and transform inputs.

### train.py
> Contains functions to orchestrate the transformers/huggingface training pipeline.