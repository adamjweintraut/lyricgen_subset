#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""gpu.py
contains functions to manage/monitor gpu memory
"""

__author__ = "Adam J. Weintraut"
__version__ = "0.0.1"

import torch
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

'''
clear_gpu():
'''
def clear_gpu():
    # collect garbage
    gc.collect()
    # clear cache with torch/cuda
    with torch.no_grad():
      torch.cuda.empty_cache()

'''
print_gpu_utilization():
'''
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

'''
print_gpu_train_summary():
'''
def print_gpu_train_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()