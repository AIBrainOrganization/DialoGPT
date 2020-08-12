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
# from pytorch_memlab import MemReporter

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
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
    # heads.append(nn.Tanh())
    self.head = nn.Sequential(*heads)
    self.init_weights()

  def forward(
      self,
      input_ids=None,
      past=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None
  ):
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
    with torch.cuda.device(transformer_outputs[0].device):
      indexes = torch.tensor(indexes).to(transformer_outputs[0].device)
      indexes = indexes.unsqueeze(-1).unsqueeze(-1)
      indexes = indexes.expand(
          (-1,) * (indexes.dim() - 1) + (transformer_outputs[0].size(-1),))

      hidden_states = transformer_outputs[0].gather(-2, indexes).squeeze(-2)

    return self.head(hidden_states)


def get_model(model_path=model_path, ctx='cpu', cachedir='~/kogpt2/',
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
      'embd_pdrop': 0.0,
      'attn_pdrop': 0.0,
      'resid_pdrop': 0.0,
      'n_head_layer': 0
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
  parser.add_argument('--model_name_or_path', type=str,
                      help='pretrained model name or path to local checkpoint')
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--max_seq_length", type=int, default=128)

  parser.add_argument("--skip_eval", action='store_true',
                      help='If true, skip evaluation.')
  parser.add_argument("--init_checkpoint", type=str)
  parser.add_argument("--train_input_file", type=str)
  parser.add_argument("--eval_input_file", type=str)
  parser.add_argument("--continue_from", type=int, default=0)

  parser.add_argument("--train_batch_size", type=int, default=4,
                      help="batch size now means per GPU per step")
  parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                      help="to increase effective batch size "
                           "and reduce synchronization")
  parser.add_argument("--eval_batch_size", type=int, default=4)
  parser.add_argument("--learning_rate", type=float, default=1e-5)
  parser.add_argument("--num_optim_steps", type=int, default=1000000,
                      help="new API specifies num update steps")
  parser.add_argument("--valid_step", type=int, default=10000,
                      help="how many optim steps between validations")
  parser.add_argument("--warmup_proportion", type=float, default=0.1)
  parser.add_argument("--warmup_steps", type=int, default=16000)

  parser.add_argument("--normalize_data", type=boolean_string, default=True)
  parser.add_argument("--fp16", type=boolean_string, default=True)
  parser.add_argument("--lr_schedule", type=str,
                      choices=['noam', 'noamwd', 'BERT', 'None'],
                      default='noam')
  parser.add_argument("--loss_scale", type=float, default=0)
  parser.add_argument("--no_token_id", type=boolean_string, default=True)

  parser.add_argument("--output_dir", type=str)
  parser.add_argument("--log_dir", type=str)
  parser.add_argument('--pbar', type=boolean_string,
                      default=True, help='turn on progress bar')

  # distributed
  parser.add_argument('--local_rank', type=int, default=-1,
                      help='for torch.distributed')
  parser.add_argument('--config', help='JSON config file')
  parser.add_argument('--target_update', type=int, default=10)

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

    optimizer = FusedAdam(parameters,
                          lr=learning_rate,
                          bias_correction=False)
    if loss_scale == 0:
      optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True,
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
  no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
  parameters = [
      {'params': [p for n, p in param_optimizer
                  if not any(nd in n for nd in no_decay)],
       'weight_decay': 0.0},
      {'params': [p for n, p in param_optimizer
                  if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  return parameters


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
        input_ids, past=past, attention_mask=attention_mask,
        position_ids=position_ids)

    outputs = self(**model_inputs)
    next_token_logits = outputs[0][:, -1, :]

    # if model has past, then set the past variable to speed up decoding
    if self._do_output_past(outputs):
      past = outputs[1]

    # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
      self.enforce_repetition_penalty_(
          next_token_logits, batch_size, 1, input_ids, repetition_penalty)

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
      next_token_logits = top_k_top_p_filtering(
          next_token_logits, top_k=top_k, top_p=top_p)
      # Min-p filtering
      next_token_logits = min_p_filtering(
          next_token_logits, min_p=min_p)
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
      sent_lengths.masked_fill_(
          is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
      # unfinished_sents is set to zero if eos in sentence
      unfinished_sents.mul_((~eos_in_sents).long())

    # stop when there is a </s> in each sentence, or if we exceed the maximul
    # length
    if unfinished_sents.max() == 0:
      break

    # extend attention_mask for new generated input if only decoder
    if self.config.is_encoder_decoder is False:
      attention_mask = torch.cat(
          [attention_mask, attention_mask.new_ones(
              (attention_mask.shape[0], 1))], dim=-1
      )

    cur_len = cur_len + 1

  # if there are different sentences lengths in the batch, some batches have to
  # be padded
  if sent_lengths.min().item() != sent_lengths.max().item():
    assert pad_token_id is not None, "`Pad_token_id` has to be defined if" \
        " batches have different lengths"
    # finished sents are filled with pad_token
    decoded = input_ids.new(
        batch_size, sent_lengths.max().item()).fill_(pad_token_id)
  else:
    decoded = input_ids

  for hypo_idx, hypo in enumerate(input_ids):
    decoded[hypo_idx, : sent_lengths[hypo_idx]
            ] = hypo[: sent_lengths[hypo_idx]]

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
    return pad_sequence(id_list, batch_first=True, padding_value=PADDING_TOKEN,
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


def eval_model_loss(policy_net, eval_dataloader, criterion, epoch, args, eos):
  policy_net.eval()
  tot_loss = 0
  tot_sample = 0
  tot_value = 0
  with torch.no_grad():
    for step, batch in enumerate(eval_dataloader):
      n_sample = batch[0].shape[0]
      loss, values = get_loss(policy_net, criterion, batch, eos, n_batch=16)
      tot_loss += loss.mean().item() * n_sample
      tot_sample += n_sample
      tot_value += values.mean().item() * n_sample
  loss = tot_loss / tot_sample
  value = tot_value / tot_sample
  print(f"\n Epoch {epoch}: Val loss {loss}, Value {value} ")
  return loss


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
  optimizer = get_optimizer(parameters, args.fp16,
                            args.loss_scale, args.learning_rate)

  with torch.cuda.device(device):
    policy_net, optimizer = amp.initialize(
        policy_net, optimizer, opt_level='O2')

  optimizer.zero_grad()
  criterion = nn.L1Loss()

  timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
  output_dir = join(args.output_dir,
                    'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate,
                                                 args.train_batch_size, n_gpu,
                                                 timestamp))
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
                     total=args.num_optim_steps, desc="training")
  else:
    pbar = None

  train_dataloader = WeightedDataLoader(
      args.train_input_file, args.train_batch_size, args.max_seq_length, 1)
  eval_dataloader = BucketingDataLoader(
      args.eval_input_file, args.eval_batch_size, args.max_seq_length)

  while True:
    tr_loss = 0.0
    nb_tr_steps = 0

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
          torch.save({k: (v.cpu() if v is not None else None)
                      for k, v in policy_net.state_dict().items()},
                     join(output_dir,
                          f'GP2-pretrain-step-{global_step}.pkl'))
          eval_loss = eval_model_loss(
              policy_net, eval_dataloader, criterion, epoch, args, eos)
          print(f'{epoch},{global_step},{eval_loss}', file=eval_logger)

        if global_step >= args.num_optim_steps:
          break

    if global_step >= args.num_optim_steps:
      break
    epoch += 1

  if pbar is not None:
    pbar.close()
  train_logger.close()
  eval_logger.close()


if __name__ == '__main__':
  random.seed(0)
  main()
