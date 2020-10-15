# coding=utf-8
# Copyright 2020 SKT AIX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import sys
import random

import gluonnlp as nlp
import requests
import torch

import numpy as np

from .model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from .utils import download as _download
from .utils import tokenizer
from .tune import load_pretrained_model
#from .tune_simple import load_pretrained_model

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 896,
    "n_embd": 768,
    "n_head": 14,
    "n_layer": 12,
    "n_positions": 128,
    "vocab_size": 50000,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1
}

random_seed = 77
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def get_pytorch_kogpt2_model(ctx='cpu', cachedir='~/kogpt2/'):
  # download model
  model_info = pytorch_kogpt2
  model_path = _download(model_info['url'],
                         model_info['fname'],
                         model_info['chksum'],
                         cachedir=cachedir)
  # download vocab
  vocab_info = tokenizer
  vocab_path = _download(vocab_info['url'],
                         vocab_info['fname'],
                         vocab_info['chksum'],
                         cachedir=cachedir)
  return get_kogpt2_model(model_path, vocab_path, ctx)


def remove_module(d):
  ret = {}
  for k, v in d.items():
    if k.startswith('module.'):
      ret[k[7:]] = v
    else:
      return d

  return ret


def get_kogpt2_model(model_file, vocab_file, ctx="cpu"):
  kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
  #d = torch.load(model_file)
  #d = remove_module(d)
  #d = load_pretrained_model(d)
  #print("d's state_dict: ")
  #for param_tensor in d.state_dict():
  #print(param_tensor, "\t", d.state_dict()[param_tensor].size())
  
  #print(kogpt2model.state_dict())
  #print("model's state_dict: ")
  for param_tensor in kogpt2model.state_dict():
    print(param_tensor, "\t", kogpt2model.state_dict()[param_tensor].size())
  
  # kogpt2model.load_state_dict(d, strict=False)
  #model_file = '/data/pytorch_kogpt2_676e9bcfa7.params'
  #d_load = load_pretrained_model(kogpt2model, model_file)
  #kogpt2model.load_state_dict(d, strict=False)
  device = torch.device(ctx)
  kogpt2model.to(device)
  kogpt2model.eval()
  vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                       mask_token=None,
                                                       sep_token=None,
                                                       cls_token=None,
                                                       unknown_token='<unk>',
                                                       padding_token='<pad>',
                                                       bos_token='<s>',
                                                       eos_token='</s>')
  return kogpt2model, vocab_b_obj
