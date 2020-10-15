import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import hashlib
import os
import sys

import gluonnlp as nlp
import requests
import torch

from model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from utils import download as _download
from utils import tokenizer
from tune import load_pretrained_model

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1
}

model_file = '/data/pytorch_kogpt2_676e9bcfa7.params'
model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))

state_dict = torch.load(model_file)
#model.load_state_dict(state_dict)
checkpoint = {'transformer.wte.weights': [50000, 768],
              'transformer.wpe.weights': [128, 128],
              'transformer.wee.weights': [7, 128], 
              'transformers.h.0.attn.c_attn.weights'
              'transformers.h.1.attn.c_attn.weights'
              'transformers.h.2.attn.c_attn.weights'
              'transformers.h.3.attn.c_attn.weights'
              'transformers.h.4.attn.c_attn.weights'
              'transformers.h.5.attn.c_attn.weights'
              'transformers.h.6.attn.c_attn.weights'
              'transformers.h.7.attn.c_attn.weights'
              'transformers.h.8.attn.c_attn.weights'
              'transformers.h.9.attn.c_attn.weights'
              'transformers.h.10.attn.c_attn.weights'
              'transformers.h.11.attn.c_attn.weights'}

torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath)
