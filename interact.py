import torch

import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from config import device_f, device_r, num_samples, MMI_temperature, top_k
from kogpt2.pytorch_kogpt2 import get_kogpt2_model
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

torch.set_grad_enabled(False)

tok_path = get_tokenizer()
tokenizer = SentencepieceTokenizer(tok_path)


VOCAB_PATH = '/home/calee/kogpt2/kogpt2_news_wiki_ko_cased_818bfa919d.spiece'
MODEL_PATH = '/home/calee/git/DialoGPT/models/output_model/' \
    'GPT2.1e-05.64.2gpu.2020-03-24160510/GP2-pretrain-step-20672.pkl'

model, vocab = get_kogpt2_model(MODEL_PATH, VOCAB_PATH, 0)

if device_f == 'cuda':
  model.half()
model.to(device_f)
model.eval()

# REVERSE_MODEL_PATH = ''
# reverse_model, _ = get_kogpt2_model(REVERSE_MODEL_PATH, VOCAB_PATH, 0)
# if device_r == 'cuda':
#   reverse_model.half()
# reverse_model.to(device_r)
# reverse_model.eval()


end_token = torch.tensor([[vocab[vocab.eos_token]]], dtype=torch.long)


def _get_response(output_token, past):
  out = torch.tensor([[]], dtype=torch.long, device=device_f)

  while True:
    output_token, past = model.forward(output_token, past=past)
    output_token = output_token[:, -1, :].float()
    indices_to_remove = output_token < torch.topk(output_token, top_k)[
        0][..., -1, None]
    output_token[indices_to_remove] = -float('Inf')
    output_token = torch.multinomial(
        F.softmax(output_token, dim=-1), num_samples=1)

    out = torch.cat((out, output_token), dim=1)

    if output_token.item() == end_token.item():
      break

  return out, past


def _score_response(output_token, correct_token):
  return 0

  # inputs = torch.cat((output_token, correct_token), dim=1)
  # mask = torch.full_like(output_token, -1, dtype=torch.long)
  # labels = torch.cat((mask, correct_token), dim=1)
  #
  # loss, _, _ = reverse_model(inputs, labels=labels)
  #
  # return -loss.float()


def append_messages(old_list: list, new_list: list, truncate_length=64):
  for message in new_list:
    if message != '':
      input_token = torch.tensor([vocab[tokenizer(message)]], dtype=torch.long)
      input_token = torch.cat((input_token, end_token), dim=1)
      old_list.append(input_token)

  if len(old_list) == 0:
    old_list.append(end_token)

  # truncate
  total_length = 0
  for i, message in enumerate(reversed(old_list)):
    total_length += message.shape[1]
    if total_length > truncate_length:
      old_list[:] = old_list[-i:]


def decode(ids):
  gen = vocab.to_tokens(ids)
  sent = ''
  for word in gen[:-1]:
    word = word.replace('▁', ' ')
    word = word.replace('<unk>', '')
    sent += word
  return sent[1:]


def generate_message(message_list: list, focus_last_message=True):
  total_input = torch.cat(message_list, dim=1).to(device_f)
  if focus_last_message:
    total_input_reversed = message_list[-1]
  else:
    total_input_reversed = torch.cat(list(reversed(message_list)), dim=1)

  past = None
  if total_input.shape[1] > 1:
    _, past = model(total_input[:, :-1])

  # results = []
  # for i in range(num_samples):
  #   result = _get_response(total_input[:, -1:], past)
  #   score = _score_response(result[0].to(
  #       device_r), total_input_reversed.to(device_r))
  #   results.append(result + (score,))
  #
  # scores = torch.stack([x[2] for x in results], dim=0)
  # winner = torch.multinomial(
  #     F.softmax(scores / MMI_temperature, dim=0), num_samples=1).item()
  # # winner = torch.argmax(scores, dim=0)
  #
  # out = results[winner][0]

  out, _ = _get_response(total_input[:, -1:], past)

  return decode(out.tolist()[0])


if __name__ == '__main__':
  my_message_list = []
  while True:
    my_message = input('usr >> ')
    append_messages(my_message_list, [my_message])
    my_response = generate_message(my_message_list)
    print('bot >>', my_response)
    append_messages(my_message_list, [my_response])
