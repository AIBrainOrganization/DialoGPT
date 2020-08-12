import os
import torch
import argparse
import logging
import datetime
import tqdm
import torch.nn as nn
import torch.nn.functional as F

from lsp_model import Adam
from data_loader import BucketingDataLoader
from gpt2_training.train_utils import boolean_string
from os.path import join
from sklearn.metrics import f1_score
from train_reinforce import get_model
from util import get_device, PADDING_TOKEN, trim, pad_sequence

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


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
    optimizer = Adam(parameters, learning_rate, max_grad_norm=1.0)
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


def min_p_filtering(logits, min_p=0, filter_value=-float("Inf")):
  if min_p > 0:
    probs = F.softmax(logits, dim=-1)
    indexes = probs < min_p
    logits[indexes] = filter_value
    # check whether this assert is right!!!!!
    assert not indexes.all(-1).any(), 'Every logits are filtered.'
  return logits


def attach_token(ids, token, pad_end=True):
  with torch.cuda.device(ids.get_device()):
    token_tensor = torch.tensor([token]).to(ids.get_device())
    id_list = []
    for id in ids:
      id_list.append(torch.cat((trim(id), token_tensor)))
    return pad_sequence(id_list, batch_first=True, padding_value=PADDING_TOKEN,
                        pad_end=pad_end)


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


def eval_model(policy_net, eval_dataloader, step, args, eos, criterion):
  policy_net.eval()
  with torch.no_grad():
    values = []
    rewards = []
    for step, batch in enumerate(eval_dataloader):
      values_batch, rewards_batch = get_values(
          policy_net, batch, eos, n_batch=16)
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

    return f1_score(true, pred, average='macro'), criterion(
        values, rewards).item()


def main():
  args = get_args()

  device = 0
  n_gpu = torch.cuda.device_count()
  args.device, args.n_gpu = device, n_gpu

  timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
  output_dir = join(args.output_dir,
                    'GPT2.{}.{}.{}gpu.{}'.format(args.learning_rate,
                                                 args.train_batch_size, n_gpu,
                                                 timestamp))
  os.makedirs(output_dir, exist_ok=True)
  log_dir = args.log_dir if args.log_dir is not None and len(
      args.log_dir) > 0 else output_dir
  eval_logger = open(join(log_dir, 'eval_log.txt'), 'a+', buffering=1)
  eval_dataloader = BucketingDataLoader(
      args.eval_input_file, args.eval_batch_size, args.max_seq_length)

  filenames = os.listdir(args.init_checkpoint)
  filenames = [f for f in filenames if f.endswith('.pkl')]

  def get_step(x):
    return int(x[18:x.index('.')])

  criterion = nn.L1Loss()

  filenames = sorted(filenames, key=get_step)

  for filename in tqdm.tqdm(filenames):
    model_path = os.path.join(args.init_checkpoint, filename)
    policy_net, vocab = get_model(model_path, device, fp16=args.fp16)
    eos = vocab[vocab.eos_token]

    step = get_step(filename)

    f1, loss = eval_model(
        policy_net, eval_dataloader, step, args, eos, criterion)
    print(f'{step},{f1},{loss}', file=eval_logger)

  eval_logger.close()


if __name__ == '__main__':
  logger = logging.getLogger('transformers.configuration_utils')
  logger.setLevel(logging.WARNING)

  main()
