'''Train script'''

import os
from os.path import join

import argparse
import logging
import datetime
import random

import tqdm
import gluonnlp as nlp
import apex.amp as amp

import torch
import torch.nn as nn
import torch.nn.functional as F

from kogpt2.model.torch_gpt2 import GPT2PreTrainedModel
from kogpt2.model.torch_gpt2 import GPT2Model, GPT2Config
from kogpt2.utils import download as _download
from kogpt2.utils import tokenizer as vocab_info
from kogpt2.pytorch_kogpt2 import remove_module
from transformers.modeling_utils import top_k_top_p_filtering

from lsp_model import Adam
from config import model_path
from data_loader import BucketingDataLoader, WeightedDataLoader
from gpt2_training.train_utils import boolean_string
from util import get_device, PADDING_TOKEN, trim, pad_sequence
from sklearn.metrics import f1_score
# from pytorch_memlab import MemReporter

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


class DQN(GPT2PreTrainedModel):
  '''DQN'''
  def __init__(self, config):
    super().__init__(config)
    self.transformer = GPT2Model(config)
    # for param in self.transformer.parameters():
    #   param.requires_grad = False

    # n_unfreeze = 6
    # for param in self.transformer.ln_f.parameters():
    #   param.requires_grad = True

    # for idx, h in enumerate(reversed(self.transformer.h)):
    #   if idx == n_unfreeze:
    #     break

    #   for param in h.parameters():
    #     param.requires_grad = True

    heads = []
    for _ in range(config.n_head_layer - 1):
      heads.append(nn.Linear(config.n_embd, config.n_embd))
      heads.append(nn.ReLU())
    heads.append(nn.Linear(config.n_embd, 1))
    self.head = nn.Sequential(*heads)
    self.init_weights()

  def forward(self,
              input_ids=None,
              past=None,
              attention_mask=None,
              token_type_ids=None,
              position_ids=None,
              head_mask=None,
              inputs_embeds=None):
    transformer_outputs = self.transformer(
        input_ids,
        past=past,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
    )
    indexes = []
    last_index = input_ids.size(-1) - 1
    for input_id in input_ids:
      index = (input_id == 0).nonzero()
      if len(index) == 0:
        indexes.append(last_index)
      else:
        indexes.append(index[0].item() - 1)
    # with torch.cuda.device(transformer_outputs[0].device):
    indexes = torch.tensor(indexes).to(transformer_outputs[0].device)
    indexes = indexes.unsqueeze(-1).unsqueeze(-1)
    indexes = indexes.expand((-1, ) * (indexes.dim() - 1) +
                             (transformer_outputs[0].size(-1), ))

    hidden_states = transformer_outputs[0].gather(-2, indexes).squeeze(-2)

    return self.head(hidden_states)


def get_model(model_path=model_path,
              ctx='cpu',
              cachedir='~/kogpt2/',
              fp16=True):
  device = torch.device(ctx)

  vocab_path = _download(vocab_info['url'],
                         vocab_info['fname'],
                         vocab_info['chksum'],
                         cachedir=cachedir)

  config = {
      "initializer_range": 0.02,
      "layer_norm_epsilon": 1e-5,
      "n_ctx": 1024,
      "n_embd": 768,
      "n_head": 12,
      "n_layer": 12,
      "n_positions": 1024,
      "vocab_size": 50000,
      'embd_pdrop': 0.1,
      'attn_pdrop': 0.1,
      'resid_pdrop': 0.1,
      'n_head_layer': 1
  }
  with torch.cuda.device(device):
    model = DQN(config=GPT2Config.from_dict(config))
    d = torch.load(model_path)
    d = remove_module(d)
    model.load_state_dict(d, strict=False)
    model.to(device)
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                   mask_token=None,
                                                   sep_token=None,
                                                   cls_token=None,
                                                   unknown_token='<unk>',
                                                   padding_token='<pad>',
                                                   bos_token='<s>',
                                                   eos_token='</s>')

    if fp16:
      logger.info('in fp16, model.half() activated')
      model.half()
  return model, vocab


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name_or_path',
                      type=str,
                      help='pretrained model name or path to local checkpoint')
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--max_seq_length", type=int, default=128)

  parser.add_argument("--skip_eval",
                      action='store_true',
                      help='If true, skip evaluation.')
  parser.add_argument("--init_checkpoint", type=str)
  parser.add_argument("--train_input_file", type=str)
  parser.add_argument("--eval_input_file", type=str)
  parser.add_argument("--continue_from", type=int, default=0)

  parser.add_argument("--train_batch_size",
                      type=int,
                      default=4,
                      help="batch size now means per GPU per step")
  parser.add_argument("--gradient_accumulation_steps",
                      type=int,
                      default=2,
                      help="to increase effective batch size "
                      "and reduce synchronization")
  parser.add_argument("--eval_batch_size", type=int, default=4)
  parser.add_argument("--learning_rate", type=float, default=1e-5)
  parser.add_argument("--num_optim_steps",
                      type=int,
                      default=1000000,
                      help="new API specifies num update steps")
  parser.add_argument("--valid_step",
                      type=int,
                      default=10000,
                      help="how many optim steps between validations")
  parser.add_argument("--warmup_proportion", type=float, default=0.1)
  parser.add_argument("--warmup_steps", type=int, default=16000)

  parser.add_argument("--normalize_data", type=boolean_string, default=True)
  parser.add_argument("--fp16", type=boolean_string, default=True)
  parser.add_argument("--lr_schedule",
                      type=str,
                      choices=['noam', 'noamwd', 'BERT', 'None'],
                      default='noam')
  parser.add_argument("--loss_scale", type=float, default=0)
  parser.add_argument("--no_token_id", type=boolean_string, default=True)

  parser.add_argument("--output_dir", type=str)
  parser.add_argument("--log_dir", type=str)
  parser.add_argument('--pbar',
                      type=boolean_string,
                      default=True,
                      help='turn on progress bar')

  # distributed
  parser.add_argument('--local_rank',
                      type=int,
                      default=-1,
                      help='for torch.distributed')
  parser.add_argument('--config', help='JSON config file')
  parser.add_argument('--target_update', type=int, default=10)

  # up sampling
  parser.add_argument("--decrease_exponent", type=float, default=0)

  # do normal parsing
  return parser.parse_args()


def get_optimizer(parameters, fp16, loss_scale, learning_rate):
  if fp16:
    logger.info('in fp16, using FusedAdam')
    try:
      from apex.optimizers import FP16_Optimizer
      from apex.optimizers import FusedAdam
    except ImportError:
      raise ImportError(
          "Please install apex from https://www.github.com/nvidia/apex "
          "to use distributed and fp16 training.")

    optimizer = FusedAdam(parameters, lr=learning_rate, bias_correction=False)
    if loss_scale == 0:
      optimizer = FP16_Optimizer(optimizer,
                                 dynamic_loss_scale=True,
                                 verbose=False)
    else:
      optimizer = FP16_Optimizer(optimizer,
                                 static_loss_scale=loss_scale,
                                 verbose=False)
  else:
    optimizer = Adam(parameters, learning_rate)
  return optimizer


def get_parameters(model):
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'ln']  # no decay for bias and LayerNorm (ln)
  parameters = [{
      'params':
      [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
      'weight_decay':
      0.0
  }, {
      'params':
      [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
      'weight_decay':
      0.0
  }]

  return parameters


@torch.no_grad()
def generate(
    self,
    input_ids=None,
    max_length=None,
    min_length=None,
    do_sample=None,
    early_stopping=None,
    num_beams=None,
    temperature=None,
    top_k=None,
    top_p=None,
    min_p=None,
    repetition_penalty=None,
    bos_token_id=None,
    pad_token_id=None,
    eos_token_id=None,
    length_penalty=None,
    no_repeat_ngram_size=None,
    num_return_sequences=None,
    attention_mask=None,
    decoder_start_token_id=None,
):
  r""" Generates sequences for models with a LM head. The method currently
  supports greedy decoding, beam-search decoding, sampling with temperature,
  sampling with top-k or nucleus sampling.

  Adapted in part from `Facebook's XLM beam search code`_.

  .. _`Facebook's XLM beam search code`:
      https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43e
      b2b2d76daeb4/src/model/transformer.py#L529


  Parameters:

      input_ids: (`optional`) `torch.LongTensor` of shape
          `(batch_size, sequence_length)`
          The sequence used as a prompt for the generation. If `None` the
          method initializes
          it as an empty `torch.LongTensor` of shape `(1,)`.

      max_length: (`optional`) int
          The max length of the sequence to be generated.  Between `min_length`
          and infinity. Default to 20.

      min_length: (`optional`) int
          The min length of the sequence to be generated.  Between 0 and
          infinity. Default to 0.

      do_sample: (`optional`) bool
          If set to `False` greedy decoding is used. Otherwise sampling is
          used. Defaults to `False` as defined in
          `configuration_utils.PretrainedConfig`.

      early_stopping: (`optional`) bool
          if set to `True` beam search is stopped when at least `num_beams`
          sentences finished per batch. Defaults to `False` as defined in
          `configuration_utils.PretrainedConfig`.

      num_beams: (`optional`) int
          Number of beams for beam search. Must be between 1 and infinity.
          1 means no beam search. Default to 1.

      temperature: (`optional`) float
          The value used to module the next token probabilities. Must be
          strictly positive. Default to 1.0.

      top_k: (`optional`) int
          The number of highest probability vocabulary tokens to keep for
          top-k-filtering. Between 1 and infinity. Default to 50.

      top_p: (`optional`) float
          The cumulative probability of parameter highest probability
          vocabulary tokens to keep for nucleus sampling. Must be between
          0 and 1. Default to 1.

      repetition_penalty: (`optional`) float
          The parameter for repetition penalty. Between 1.0 and infinity.
          1.0 means no penalty. Default to 1.0.

      pad_token_id: (`optional`) int
          Padding token. Default to specicic model pad_token_id or None if it
          does not exist.

      bos_token_id: (`optional`) int
          BOS token. Defaults to bos_token_id as defined in the models config.

      pad_token_id: (`optional`) int
          Pad token. Defaults to pad_token_id as defined in the models config.

      eos_token_ids: (`optional`) int or list of int
          End of sequence token or list of tokens to stop the generation.
          Default to eos_token_ids as defined in the models config.

      length_penalty: (`optional`) float
          Exponential penalty to the length. Default to 1.

      no_repeat_ngram_size: (`optional`) int
          If set to int > 0, all ngrams of size `no_repeat_ngram_size` can
          only occur once.

      num_return_sequences: (`optional`) int
          The number of independently computed returned sequences for each
          element in the batch. Default to 1.

      attention_mask (`optional`) obj: `torch.LongTensor` of same shape as
          `input_ids`
          Mask to avoid performing attention on padding token indices.
          Mask values selected in ``[0, 1]``:
          ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
          Defaults to `None`.

      `What are attention masks? <../glossary.html#attention-mask>`__

      decoder_start_token_id=None: (`optional`) int
          If an encoder-decoder model starts decoding with a different token
          than BOS.
          Defaults to `None` and is changed to `BOS` later.

  Return:

      output: `torch.LongTensor` of shape `(batch_size * num_return_sequences,
          equence_length)`
          sequence_length is either equal to max_length or shorter if all
          batches finished early due to the `eos_token_id`

  Examples::
      # Initialize tokenizer
      tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
      # Download model and configuration from S3 and cache.
      model = AutoModelWithLMHead.from_pretrained('distilgpt2')
      outputs = model.generate(max_length=40)  # do greedy decoding
      print('Generated: {}'.format(tokenizer.decode(outputs[0],
          skip_special_tokens=True)))
      # Initialize tokenizer
      tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
      # Download model and configuration from S3 and cache.
      model = AutoModelWithLMHead.from_pretrained('openai-gpt')
      input_context = 'The dog'
      # encode input context
      input_ids = tokenizer.encode(input_context, return_tensors='pt')
      # generate 3 independent sequences using beam search decoding (5 beams)
      # with sampling from initial context 'The dog'
      outputs = model.generate(input_ids=input_ids, num_beams=5,
          num_return_sequences=3, temperature=1.5)
      for i in range(3): #  3 output sequences were generated
          print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i],
              skip_special_tokens=True)))
      # Initialize tokenizer
      tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
      # Download model and configuration from S3 and cache.
      model = AutoModelWithLMHead.from_pretrained('distilgpt2')
      input_context = 'The dog'
      # encode input context
      input_ids = tokenizer.encode(input_context, return_tensors='pt')
      # 3 generate sequences using by sampling
      outputs = model.generate(input_ids=input_ids, max_length=40,
          temperature=0.7, num_return_sequences=3)
      for i in range(3): #  3 output sequences were generated
          print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i],
              skip_special_tokens=True)))

      # Initialize tokenizer
      tokenizer = AutoTokenizer.from_pretrained('ctrl')
      # Download model and configuration from S3 and cache.
      model = AutoModelWithLMHead.from_pretrained('ctrl')
      # "Legal" is one of the control codes for ctrl
      input_context = 'Legal My neighbor is'
      # encode input context
      input_ids = tokenizer.encode(input_context, return_tensors='pt')
      # generate sequences
      outputs = model.generate(input_ids=input_ids, max_length=50,
          temperature=0.7, repetition_penalty=1.2)
      print('Generated: {}'.format(tokenizer.decode(outputs[0],
          skip_special_tokens=True)))

  """

  # We cannot generate if the model does not have a LM head
  if self.get_output_embeddings() is None:
    raise AttributeError(
        "You tried to generate sequences with a model that does not have a LM"
        " Head."
        "Please use another model class (e.g. `OpenAIGPTLMHeadModel`,"
        " `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`,"
        " `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`,"
        " `BartForConditionalGeneration` )")

  max_length = max_length if max_length is not None else self.config.max_length
  min_length = min_length if min_length is not None else self.config.min_length
  do_sample = do_sample if do_sample is not None else self.config.do_sample
  early_stopping = (early_stopping if early_stopping is not None else
                    self.config.early_stopping)
  num_beams = num_beams if num_beams is not None else self.config.num_beams
  temperature = (temperature
                 if temperature is not None else self.config.temperature)
  top_k = top_k if top_k is not None else self.config.top_k
  top_p = top_p if top_p is not None else self.config.top_p
  repetition_penalty = (repetition_penalty if repetition_penalty is not None
                        else self.config.repetition_penalty)
  bos_token_id = (bos_token_id
                  if bos_token_id is not None else self.config.bos_token_id)
  pad_token_id = (pad_token_id
                  if pad_token_id is not None else self.config.pad_token_id)
  eos_token_id = (eos_token_id
                  if eos_token_id is not None else self.config.eos_token_id)
  length_penalty = (length_penalty if length_penalty is not None else
                    self.config.length_penalty)
  no_repeat_ngram_size = (no_repeat_ngram_size if no_repeat_ngram_size
                          is not None else self.config.no_repeat_ngram_size)
  num_return_sequences = (num_return_sequences if num_return_sequences
                          is not None else self.config.num_return_sequences)
  decoder_start_token_id = (decoder_start_token_id if decoder_start_token_id
                            is not None else bos_token_id)

  if input_ids is not None:
    batch_size = input_ids.shape[0]  # overriden by the input batch_size
  else:
    batch_size = 1

  assert isinstance(
      max_length, int
  ) and max_length > 0, "`max_length` should be a strictly positive integer."
  assert isinstance(
      min_length,
      int) and min_length >= 0, "`min_length` should be a positive integer."
  assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
  assert isinstance(early_stopping,
                    bool), "`early_stopping` should be a boolean."
  assert isinstance(
      num_beams, int
  ) and num_beams > 0, "`num_beams` should be a strictly positive integer."
  assert temperature > 0, "`temperature` should be strictly positive."
  assert isinstance(
      top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
  assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
  assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
  assert input_ids is not None or (
      isinstance(bos_token_id, int) and bos_token_id >= 0
  ), "If input_ids is not defined, `bos_token_id` should be a positive" \
      " integer."
  assert pad_token_id is None or (
      isinstance(pad_token_id, int) and
      (pad_token_id >= 0)), "`pad_token_id` should be a positive integer."
  assert (
      decoder_start_token_id is not None
      or self.config.is_encoder_decoder is False
  ), "`decoder_start_token_id` has to be defined if model is encoder-decoder" \
      " model"
  assert (eos_token_id is None) or (
      isinstance(eos_token_id, int) and
      (eos_token_id >= 0)), "`eos_token_id` should be a positive integer."
  assert length_penalty > 0, "`length_penalty` should be strictly positive."
  assert (isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
          ), "`no_repeat_ngram_size` should be a positive integer."
  assert (isinstance(num_return_sequences, int) and num_return_sequences > 0
          ), "`num_return_sequences` should be a strictly positive integer."

  if input_ids is None:
    assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
        "you should either supply a context to complete as `input_ids` input "
        "or a `bos_token_id` (integer >= 0) as a first token to start the"
        " generation.")
    input_ids = torch.full(
        (batch_size, 1),
        bos_token_id,
        dtype=torch.long,
        device=next(self.parameters()).device,
    )
  else:
    assert input_ids.dim(
    ) == 2, "Input prompt should be of shape (batch_size, sequence length)."

  # not allow to duplicate outputs when greedy decoding
  if do_sample is False:
    if num_beams == 1:
      # no_beam_search greedy generation conditions
      assert (
          num_return_sequences == 1
      ), "Greedy decoding will always produce the same output for" \
          " num_beams == 1 and num_return_sequences > 1. Please set" \
          " num_return_sequences = 1"

    else:
      # beam_search greedy generation conditions
      assert (
          num_beams >= num_return_sequences
      ), "Greedy beam search decoding cannot return more sequences than it" \
          " has beams. Please set num_beams >= num_return_sequences"

  # create attention mask if necessary
  # TODO (PVP): this should later be handled by the forward fn() in each model
  # in the future see PR 3140
  if (attention_mask is None) and (pad_token_id
                                   is not None) and (pad_token_id
                                                     in input_ids):
    attention_mask = input_ids.ne(pad_token_id).long()
  elif attention_mask is None:
    attention_mask = input_ids.new_ones(input_ids.shape)

  # set pad_token_id to eos_token_id if not set. Important that this is done
  # after attention_mask is created
  if pad_token_id is None and eos_token_id is not None:
    logger.warning(
        "Setting `pad_token_id` to {} (first `eos_token_id`) to generate"
        " sequence".format(eos_token_id))
    pad_token_id = eos_token_id

  # current position and vocab size
  vocab_size = self.config.vocab_size

  # set effective batch size and effective batch multiplier according to
  # do_sample
  if do_sample:
    effective_batch_size = batch_size * num_return_sequences
    effective_batch_mult = num_return_sequences
  else:
    effective_batch_size = batch_size
    effective_batch_mult = 1

  # Expand input ids if num_beams > 1 or num_return_sequences > 1
  if num_return_sequences > 1 or num_beams > 1:
    input_ids_len = input_ids.shape[-1]
    input_ids = input_ids.unsqueeze(1).expand(batch_size,
                                              effective_batch_mult * num_beams,
                                              input_ids_len)
    attention_mask = attention_mask.unsqueeze(1).expand(
        batch_size, effective_batch_mult * num_beams, input_ids_len)

    input_ids = input_ids.contiguous().view(
        effective_batch_size * num_beams, input_ids_len
    )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
    attention_mask = attention_mask.contiguous().view(
        effective_batch_size * num_beams, input_ids_len
    )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

  if self.config.is_encoder_decoder:
    assert bos_token_id is not None, "Encoder Decoder Models need" \
        " to have a bos_token_id"
    assert hasattr(
        self, "get_encoder"
    ), "{} should have a 'get_encoder' function defined".format(self)
    assert callable(self.get_encoder), "{} should be a method".format(
        self.get_encoder)

    # get encoder and store encoder outputs
    encoder = self.get_encoder()

    encoder_outputs = encoder(input_ids, attention_mask=attention_mask)

    # create empty decoder_input_ids
    input_ids = torch.full(
        (effective_batch_size * num_beams, 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=next(self.parameters()).device,
    )
    cur_len = 1
  else:
    encoder_outputs = None
    cur_len = input_ids.shape[-1]

  if num_beams > 1:
    output = self._generate_beam_search(
        input_ids,
        cur_len=cur_len,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        early_stopping=early_stopping,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
        eos_token_id=eos_token_id,
        batch_size=effective_batch_size,
        num_return_sequences=num_return_sequences,
        length_penalty=length_penalty,
        num_beams=num_beams,
        vocab_size=vocab_size,
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
    )
  else:
    output = _generate_no_beam_search(
        self,
        input_ids,
        cur_len=cur_len,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
        eos_token_id=eos_token_id,
        batch_size=effective_batch_size,
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
    )

  return output


def min_p_filtering(logits, min_p=0, filter_value=-float("Inf")):
  if min_p > 0:
    probs = F.softmax(logits, dim=-1)
    indexes = probs < min_p
    logits[indexes] = filter_value
    # check whether this assert is right!!!!!
    assert not indexes.all(-1).any(), 'Every logits are filtered.'
  return logits


def _generate_no_beam_search(
    self,
    input_ids,
    cur_len,
    max_length,
    min_length,
    do_sample,
    temperature,
    top_k,
    top_p,
    min_p,
    repetition_penalty,
    no_repeat_ngram_size,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    decoder_start_token_id,
    batch_size,
    encoder_outputs,
    attention_mask,
):
  """ Generate sequences for each example without beam search (num_beams == 1).
      All returned sequence are generated independantly.
  """
  # length of generated sentences / unfinished sentences
  unfinished_sents = input_ids.new(batch_size).fill_(1)
  sent_lengths = input_ids.new(batch_size).fill_(max_length)

  past = encoder_outputs
  # defined for encoder-decoder models, None for decoder-only models

  position_ids = []
  while cur_len < max_length:
    # create position_ids assuming pads are in front.
    # should consider past
    len_input_ids = input_ids.size(-1)
    if len(position_ids) == 0:
      for input_id in input_ids:
        indexes = (input_id == pad_token_id).nonzero()
        if len(indexes) == 0:
          n_zeros = 0
        else:
          n_zeros = indexes[-1].item() + 1
        position_ids.append(
            torch.arange(-n_zeros, len_input_ids - n_zeros, dtype=torch.long))

      position_ids = torch.stack(position_ids, dim=0)
      position_ids[position_ids < 0] = 0
      with torch.cuda.device(input_ids.get_device()):
        position_ids = position_ids.to(input_ids.get_device())
    else:
      position_ids = position_ids[:, -1:] + 1

    model_inputs = self.prepare_inputs_for_generation(
        input_ids,
        past=past,
        attention_mask=attention_mask,
        position_ids=position_ids)

    outputs = self(**model_inputs)
    next_token_logits = outputs[0][:, -1, :]

    # if model has past, then set the past variable to speed up decoding
    if self._do_output_past(outputs):
      past = outputs[1]

    # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
      self.enforce_repetition_penalty_(next_token_logits, batch_size, 1,
                                       input_ids, repetition_penalty)

    if no_repeat_ngram_size > 0:
      # calculate a list of banned tokens to prevent repetitively generating
      # the same ngrams
      # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9
      # e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
      raise Exception('calc_banned_tokens?')
      # banned_tokens = calc_banned_tokens(
      #     input_ids, batch_size, no_repeat_ngram_size, cur_len)
      # for batch_idx in range(batch_size):
      #   next_token_logits[batch_idx,
      #                     banned_tokens[batch_idx]] = -float("inf")
    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
      next_token_logits[:, eos_token_id] = -float("inf")

    if do_sample:
      # Temperature (higher temperature => more likely to sample low
      # probability tokens)
      if temperature != 1.0:
        next_token_logits = next_token_logits / temperature

      # Top-p/top-k filtering
      next_token_logits = top_k_top_p_filtering(next_token_logits,
                                                top_k=top_k,
                                                top_p=top_p)
      # Min-p filtering
      next_token_logits = min_p_filtering(next_token_logits, min_p=min_p)
      # Sample
      probs = F.softmax(next_token_logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
      # Greedy decoding
      next_token = torch.argmax(next_token_logits, dim=-1)

    # update generations and finished sentences
    if eos_token_id is not None:
      # pad finished sentences if eos_token_id exist
      tokens_to_add = next_token * unfinished_sents + \
          (pad_token_id) * (1 - unfinished_sents)
    else:
      tokens_to_add = next_token

    input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

    if eos_token_id is not None:
      eos_in_sents = tokens_to_add == eos_token_id
      # if sentence is unfinished and the token to add is eos, sent_lengths is
      # filled with current length
      is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(
          eos_in_sents.long()).bool()
      sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos,
                                cur_len + 1)
      # unfinished_sents is set to zero if eos in sentence
      unfinished_sents.mul_((~eos_in_sents).long())

    # stop when there is a </s> in each sentence, or if we exceed the maximul
    # length
    if unfinished_sents.max() == 0:
      break

    # extend attention_mask for new generated input if only decoder
    if self.config.is_encoder_decoder is False:
      tensors = [
          attention_mask,
          attention_mask.new_ones((attention_mask.shape[0], 1))
      ]
      attention_mask = torch.cat(tensors, dim=-1)

    cur_len = cur_len + 1

  # if there are different sentences lengths in the batch, some batches have to
  # be padded
  if sent_lengths.min().item() != sent_lengths.max().item():
    assert pad_token_id is not None, "`Pad_token_id` has to be defined if" \
        " batches have different lengths"
    # finished sents are filled with pad_token
    decoded = input_ids.new(batch_size,
                            sent_lengths.max().item()).fill_(pad_token_id)
  else:
    decoded = input_ids

  for hypo_idx, hypo in enumerate(input_ids):
    decoded[hypo_idx, :sent_lengths[hypo_idx]] = hypo[:sent_lengths[hypo_idx]]

  return decoded


# _, model, reverse_model, _ = load(
#     vocab_path, model_path, reverse_model_path, 0)
#
# model._generate_no_beam_search = types.MethodType(
#     _generate_no_beam_search, model)
#
# min_p = min_p_alpha / model.lm_head.out_features


def attach_token(ids, token, pad_end=True):
  with torch.cuda.device(ids.get_device()):
    token_tensor = torch.tensor([token]).to(ids.get_device())
    id_list = []
    for id in ids:
      id_list.append(torch.cat((trim(id), token_tensor)))
    return pad_sequence(id_list,
                        batch_first=True,
                        padding_value=PADDING_TOKEN,
                        pad_end=pad_end)


def get_loss(policy_net, criterion, batch, eos, n_batch=32, GAMMA=0.999):
  # input_ids contains both state and action
  input_ids, rewards = batch
  with torch.cuda.device(get_device(policy_net)):
    input_ids = input_ids.to(get_device(policy_net))

    state_action_values = policy_net(attach_token(input_ids, eos))

  with torch.no_grad():
    with torch.cuda.device(state_action_values.get_device()):
      rewards = rewards.to(state_action_values.get_device()).reshape(-1, 1)

  loss = criterion(state_action_values, rewards.float())

  if loss.dim() != 0:
    loss = loss.mean()

  return loss, state_action_values


def get_values(policy_net, batch, eos, n_batch=32, GAMMA=0.999):
  # input_ids contains both state and action
  input_ids, rewards = batch
  device = get_device(policy_net)
  with torch.cuda.device(device):
    input_ids = input_ids.to(device)
    state_action_values = policy_net(attach_token(input_ids, eos))

  with torch.no_grad():
    with torch.cuda.device(device):
      rewards = rewards.to(device).reshape(-1, 1)

  return state_action_values, rewards


def eval_model(policy_net, eval_dataloader, args, eos, criterion):
  policy_net.eval()
  with torch.no_grad():
    values = []
    rewards = []
    for batch in eval_dataloader:
      values_batch, rewards_batch = get_values(policy_net,
                                               batch,
                                               eos,
                                               n_batch=16)
      values.append(values_batch.cpu().float())
      rewards.append(rewards_batch.cpu().float())

    values = torch.cat(values).squeeze(1)
    rewards = torch.cat(rewards).squeeze(1)

    pred = (values < -1 / 3).int()
    pred[values > 1 / 3] = 2

    true = (rewards < -1 / 3).int()
    true[rewards > 1 / 3] = 2

    # pred = (values < 0).int()
    # true = (rewards < 0).int()

    return f1_score(true, pred, average='macro'), criterion(values,
                                                            rewards).item()


def main():
  args = get_args()

  global model_path
  if args.init_checkpoint is not None:
    model_path = args.init_checkpoint

  device = 0
  n_gpu = torch.cuda.device_count()
  args.device, args.n_gpu = device, n_gpu

  assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
      'batch size % gradient accumulation steps != 0!'
  args.train_batch_size = args.train_batch_size // \
      args.gradient_accumulation_steps

  policy_net, vocab = get_model(model_path, ctx=device, fp16=args.fp16)

  eos = vocab[vocab.eos_token]

  parameters = get_parameters(policy_net)
  optimizer = get_optimizer(parameters, args.fp16, args.loss_scale,
                            args.learning_rate)

  with torch.cuda.device(device):
    policy_net, optimizer = amp.initialize(policy_net,
                                           optimizer,
                                           opt_level='O2')

  optimizer.zero_grad()
  criterion = nn.L1Loss()

  timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
  output_dir = join(
      args.output_dir,
      'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate, args.train_batch_size,
                                   n_gpu, timestamp))
  os.makedirs(output_dir, exist_ok=True)
  log_dir = args.log_dir if args.log_dir is not None and len(
      args.log_dir) > 0 else output_dir
  train_logger = open(join(log_dir, 'train_log.txt'), 'a+', buffering=1)
  eval_logger = open(join(log_dir, 'eval_log.txt'), 'a+', buffering=1)

  epoch = 0
  global_step = 0  # optimizer step
  step = 0

  if args.pbar:
    pbar = tqdm.tqdm(initial=global_step,
                     total=args.num_optim_steps,
                     desc="training")
  else:
    pbar = None
  eval_dataloader = BucketingDataLoader(args.eval_input_file,
                                        args.eval_batch_size,
                                        args.max_seq_length)

  exponent = 1

  while True:
    tr_loss = 0.0
    nb_tr_steps = 0

    train_dataloader = WeightedDataLoader(args.train_input_file,
                                          args.train_batch_size,
                                          args.max_seq_length, exponent)

    for batch in train_dataloader:
      policy_net.train()

      loss, _ = get_loss(policy_net, criterion, batch, eos)
      if args.fp16:
        optimizer.backward(loss)
      else:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
        # loss.backward()

      tr_loss += float(loss.item())
      nb_tr_steps += 1
      mean_loss = tr_loss / nb_tr_steps

      step += 1

      if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        if pbar is not None:
          pbar.set_postfix_str(f'loss: {mean_loss:0.4e} epoch: {epoch}')
          pbar.update(1)
        print(f'{epoch},{global_step},{mean_loss}', file=train_logger)

        if global_step % args.valid_step == 0:
          # save to cpu tensors
          torch.save(
              {
                  k: (v.cpu() if v is not None else None)
                  for k, v in policy_net.state_dict().items()
              }, join(output_dir, f'GP2-pretrain-step-{global_step}.pkl'))
          f1, eval_loss = eval_model(policy_net, eval_dataloader, args, eos,
                                     criterion)
          print(f'{epoch},{global_step},{f1},{eval_loss}', file=eval_logger)
          print(f'{epoch},{global_step},{f1},{eval_loss}')

        if global_step >= args.num_optim_steps:
          break

    if global_step >= args.num_optim_steps:
      break
    exponent = max(exponent - args.decrease_exponent, 0)
    epoch += 1

  if pbar is not None:
    pbar.close()
  train_logger.close()
  eval_logger.close()


if __name__ == '__main__':
  random.seed(0)
  main()
