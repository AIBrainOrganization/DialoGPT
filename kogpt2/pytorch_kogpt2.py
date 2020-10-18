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

import gluonnlp as nlp
import requests
import torch

from .model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from .utils import download as _download
from .utils import tokenizer

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
    "n_positions": 1024,
    "vocab_size": 50000,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'output_past' : False  # model test를 위한 임시 변수 추가

}


def get_pytorch_kogpt2_model(ctx='cpu', cachedir='~/kogpt2/'):
  # download model
  model_path = None
  vocab_path = None
#  model_info = pytorch_kogpt2
#  model_path = _download(model_info['url'],
#                         model_info['fname'],
#                         model_info['chksum'],
#                         cachedir=cachedir)
#  # download vocab
#  vocab_info = tokenizer
#  vocab_path = _download(vocab_info['url'],
#                         vocab_info['fname'],
#                         vocab_info['chksum'],
#                         cachedir=cachedir)
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
  if model_file is not None:
    d = torch.load(model_file)
    d = remove_module(d)
    kogpt2model.load_state_dict(d)
  device = torch.device(ctx)
  kogpt2model.to(device)
  kogpt2model.eval()
  vocab_b_obj = None
  if vocab_file is not None:
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         mask_token=None,
                                                         sep_token=None,
                                                         cls_token=None,
                                                         unknown_token='<unk>',
                                                         padding_token='<pad>',
                                                         bos_token='<s>',
                                                         eos_token='</s>')
    # 감정 토큰 추가
    emotion_list = ['neutral', 'happiness', 'surprise', 'disgust', 'angry', 'fear', 'sadness'] # unused 6 ~ 12와 같음
  
    # '<neutral>' : 6
    # '<happiness>' : 7
    # '<surprise>' : 8
    # '<disgust>' : 9
    # '<angry>' : 10
    # '<fear>' : 11
    # '<sadness>' : 12
  
    for i, emo in enumerate(emotion_list):
      vocab_b_obj.token_to_idx['<{}>'.format(emo)] = vocab_b_obj.token_to_idx["<unused{}>".format(i)]
      del vocab_b_obj.token_to_idx["<unused{}>".format(i)]
      vocab_b_obj.idx_to_token[i+6] = '<{}>'.format(emo)  # idx -> token list 수정
  return kogpt2model, vocab_b_obj
