import os
import torch
import argparse
import logging
import datetime
import tqdm
import gluonnlp as nlp
import torch.nn as nn
import torch.nn.functional as F

from kogpt2.model.torch_gpt2 import GPT2PreTrainedModel
from kogpt2.model.torch_gpt2 import GPT2Model, GPT2Config
from kogpt2.utils import download as _download
from kogpt2.utils import tokenizer as vocab_info
from kogpt2.pytorch_kogpt2 import remove_module
from lsp_model import Adam
from interact import load, PADDING_TOKEN, _score_response
from config import vocab_path, model_path, reverse_model_path
from config import top_k, top_p, ALPHA, BETA, num_samples
from data_loader import BucketingDataLoader
from gpt2_training.train_utils import boolean_string
from os.path import join
from torch.nn.utils.rnn import pad_sequence
from util import get_device

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class DQN(GPT2PreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.transformer = GPT2Model(config)
    self.head = nn.Linear(config.n_embd, 1, bias=False)
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
    hidden_states = transformer_outputs[0][:, -1, :]

    return self.head(hidden_states)


def get_model(ctx='cpu', cachedir='~/kogpt2/', fp16=True):
  model_info = {
      'url':
      'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
      'fname': 'pytorch_kogpt2_676e9bcfa7.params',
      'chksum': '676e9bcfa7'
  }

  model_path = _download(model_info['url'],
                         model_info['fname'],
                         model_info['chksum'],
                         cachedir=cachedir)

  vocab_path = _download(vocab_info['url'],
                         vocab_info['fname'],
                         vocab_info['chksum'],
                         cachedir=cachedir)

  config = {
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
  model = DQN(config=GPT2Config.from_dict(config))
  d = torch.load(model_path)
  d = remove_module(d)
  model.load_state_dict(d, strict=False)
  device = torch.device(ctx)
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
                      choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
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
                          bias_correction=False,
                          max_grad_norm=1.0)
    if loss_scale == 0:
      optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True,
                                 verbose=False)
    else:
      optimizer = FP16_Optimizer(optimizer,
                                 static_loss_scale=loss_scale,
                                 verbose=False)
  else:
    optimizer = Adam(optimizer_grouped_parameters, learning_rate,
                     max_grad_norm=1.0)
  return optimizer


def get_parameters(model):
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
  parameters = [
      {'params': [p for n, p in param_optimizer
                  if not any(nd in n for nd in no_decay)],
       'weight_decay': 0.01},
      {'params': [p for n, p in param_optimizer
                  if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  return parameters


_, model, reverse_model, _ = load(
    vocab_path, model_path, reverse_model_path, 0)


def trim(sequence):
  pad_index = (sequence == PADDING_TOKEN).nonzero()
  if sequence.dim() == 1:
    if len(pad_index) == 0:
      end_index = len(sequence)
    else:
      end_index = pad_index[0].item()
    return sequence[:end_index]
  else:
    if len(pad_index) == 0:
      end_index = sequence.shape[1]
    else:
      end_index = pad_index[0][1].item()
    return sequence[:, :end_index]


def reverse(ids, eos):
  eos_tensor = torch.tensor([eos]).to(ids.get_device())
  id_list = []
  for id in ids:
    eos_index = (id == eos).nonzero()
    sequences = []
    start_index = 0
    for i in eos_index:
      if i == eos_index[0]:
        end_index = i
      else:
        end_index = i + 1
      sequences.append(id[start_index:end_index])
      start_index = i + 1
    sequences.append(eos_tensor)
    sequences.append(trim(id[start_index:]))

    id_list.append(torch.cat(tuple(reversed(sequences))))

  return pad_sequence(id_list, batch_first=True, padding_value=PADDING_TOKEN)


def attach_token(ids, token):
  token_tensor = torch.tensor([token]).to(ids.get_device())
  id_list = []
  for id in ids:
    id_list.append(torch.cat((trim(id), token_tensor)))
  return pad_sequence(id_list, batch_first=True, padding_value=PADDING_TOKEN)


def get_next_actions(target_net, next_states, eos):
  # should return batches
  next_states = next_states.to(get_device(model))
  sequences = []
  n_batch = 4
  assert len(next_states) % n_batch == 0, \
      'Num batch is not multiple of n_batch.'
  for i in tqdm.tqdm(range(0, len(next_states), n_batch), desc='actions'):
    next_state = attach_token(next_states[i:i + n_batch], eos)
    outputs = model.generate(
        input_ids=next_state, min_length=next_state.shape[1] + 5,
        max_length=next_state.shape[1] + 40,
        num_return_sequences=num_samples,
        top_k=top_k, top_p=top_p, do_sample=True, repetition_penalty=1.2,
        pad_token_id=PADDING_TOKEN, eos_token_id=eos)

    outputs = outputs.reshape(-1, num_samples, outputs.shape[1])

    for j in range(n_batch):
      samples = outputs[j, :, :]
      scores = []
      for output in samples:
        output = output.unsqueeze(0)
        try:
          output = output[:, :output[0].tolist().index(
              eos) + 1]
        except:
          pass
        sentence = next_state[j:j + 1]
        scores.append(_score_response(
            attach_token(sentence, eos), attach_token(
                reverse(sentence, eos), eos), output,
            model, reverse_model, target_net))
      scores = torch.stack(scores, dim=0)

      # 가장 점수가 높은 문장을 선택합니다.
      winner = torch.argmax(scores).item()
      sequences.append(samples[winner])
  return pad_sequence(sequences, batch_first=True, padding_value=PADDING_TOKEN)


def concat(a, b, eos):
  eos = torch.tensor([eos]).to(a.get_device())
  sequences = []
  for i in range(len(a)):
    zero_index = (a[i] == 0).nonzero()
    if len(zero_index) == 0:
      a_i = a[i]
    else:
      a_i = a[i][:zero_index[0].item()]

    zero_index = (b[i] == 0).nonzero()
    if len(zero_index) == 0:
      b_i = b[i]
    else:
      b_i = b[i][:zero_index[0].item()]

    sequences.append(torch.cat((a_i, eos, b_i)))

  return pad_sequence(sequences, batch_first=True, padding_value=PADDING_TOKEN)


def get_loss(policy_net, target_net, criterion, batch, eos, GAMMA=0.999):
  # input_ids contains both state and action
  input_ids, next_states, rewards = batch

  input_ids = input_ids.to(get_device(policy_net))

  state_action_values = policy_net(attach_token(input_ids, eos))

  with torch.no_grad():
    input_ids = input_ids.to(get_device(target_net))
    next_states = next_states.to(get_device(target_net))

    next_states = concat(input_ids, next_states, eos)

    next_actions = get_next_actions(
        target_net, next_states, eos).to(get_device(target_net))

    next_state_values = target_net(next_actions)

    rewards = rewards.to(next_state_values.get_device()).reshape(-1, 1)
    expected_state_action_values = next_state_values * GAMMA + rewards

  loss = criterion(state_action_values, expected_state_action_values)

  if loss.dim() != 0:
    loss = loss.mean()

  return loss


def eval_model_loss(policy_net, target_net, eval_dataloader, epoch, args, eos):
  policy_net.eval()
  tot_loss = []
  tot_sample = []
  with torch.no_grad():
    for step, batch in enumerate(eval_dataloader):
      batch = tuple(t.to(args.device) for t in batch)
      input_ids, position_ids, token_ids, label_ids, *_ = batch
      n_sample = input_ids.shape[0]
      import pdb
      pdb.set_trace()
      loss = get_loss(policy_net, target_net, criterion, batch, eos)
      tot_loss.append(loss.mean().item() * n_sample)
      tot_sample.append(n_sample)
  print(f"\n Epoch {epoch}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} ")
  return np.sum(tot_loss) / np.sum(tot_sample)


def main():
  args = get_args()

  device = 1
  n_gpu = torch.cuda.device_count()
  args.device, args.n_gpu = device, n_gpu

  assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
      'batch size % gradient accumulation steps != 0!'
  args.train_batch_size = args.train_batch_size // \
      args.gradient_accumulation_steps

  policy_net, vocab = get_model(device)
  target_net = get_model(device)[0]
  target_net.eval()

  eos = vocab[vocab.eos_token]

  parameters = get_parameters(policy_net)
  optimizer = get_optimizer(parameters, args.fp16,
                            args.loss_scale, args.learning_rate)
  optimizer.zero_grad()
  criterion = nn.SmoothL1Loss()

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
                     total=args.num_optim_steps, desc=f"training")
  else:
    pbar = None

  train_dataloader = BucketingDataLoader(
      args.train_input_file, args.train_batch_size, args.max_seq_length)
  eval_dataloader = BucketingDataLoader(
      args.eval_input_file, args.eval_batch_size, args.max_seq_length)

  while True:
    tr_loss = 0.0
    nb_tr_steps = 0

    for batch in train_dataloader:
      policy_net.train()
      loss = get_loss(policy_net, target_net, criterion, batch, eos)
      if args.fp16:
        optimizer.backward(loss)
      else:
        loss.backward()

      tr_loss += float(loss.item())
      nb_tr_steps += 1
      mean_loss = tr_loss / nb_tr_steps

      step += 1
      if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        if pbar is not None:
          pbar.set_postfix_str(f'loss: {mean_loss} epoch: {epoch}')
          pbar.update(1)
        print(f'{epoch},{global_step},{mean_loss}', file=train_logger)

        if global_step % args.valid_step == 0:
          # save to cpu tensors
          torch.save({k: (v.cpu() if v is not None else None)
                      for k, v in model.state_dict().items()},
                     join(output_dir,
                          f'GP2-pretrain-step-{global_step}.pkl'))
          eval_loss = eval_model_loss(
              policy_net, target_net, eval_dataloader, epoch, args, eos)
          print(f'{epoch},{global_step},{eval_loss}', file=eval_logger)

        if global_step % args.target_update == 0:
          target_net.load_state_dict(policy_net.state_dict())

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
  main()
