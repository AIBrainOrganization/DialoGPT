import torch
import argparse

import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from config import device_f, device_r, num_samples
from config import MMI_temperature, top_k, top_p, ALPHA
from kogpt2.pytorch_kogpt2 import get_kogpt2_model
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

PADDING_TOKEN = 0

torch.set_grad_enabled(False)

tok_path = get_tokenizer()
tokenizer = SentencepieceTokenizer(tok_path)


def load(vocab_path, model_path, reverse_path):
  # VOCAB_PATH = '/home/calee/kogpt2/kogpt2_news_wiki_ko_cased_818bfa919d.spiece'
  # MODEL_PATH = '/home/calee/git/DialoGPT/models/output_model/' \
  #     'GPT2.1e-05.64.2gpu.2020-03-24160510/GP2-pretrain-step-20672.pkl'

  model, vocab = get_kogpt2_model(model_path, vocab_path, 0)

  if device_f == 'cuda':
    model.half()
  model.to(device_f)
  model.eval()

  # REVERSE_MODEL_PATH = '/home/calee/git/DialoGPT/models/output_model/' \
  #     'GPT2.1e-05.64.2gpu.2020-03-26162233/GP2-pretrain-step-8704.pkl'
  reverse_model, _ = get_kogpt2_model(reverse_path, vocab_path, 0)
  if device_r == 'cuda':
    reverse_model.half()
  reverse_model.to(device_r)
  reverse_model.eval()

  end_token = torch.tensor([[vocab[vocab.eos_token]]], dtype=torch.long)

  return vocab, model, reverse_model, end_token


# def _get_response(output_token, past, end_token):
#   out = torch.tensor([[]], dtype=torch.long, device=device_f)
#
#   # check model.generate exist and working!!!
#   while True:
#     output_token, past = model.forward(output_token, past=past)
#     output_token = output_token[:, -1, :].float()
#     indices_to_remove = output_token < torch.topk(output_token, top_k)[
#         0][..., -1, None]
#     output_token[indices_to_remove] = -float('Inf')
#     output_token = torch.multinomial(
#         F.softmax(output_token, dim=-1), num_samples=1)
#
#     out = torch.cat((out, output_token), dim=1)
#
#     if output_token.item() == end_token.item():
#       break
#
#   return out, past


def _score_response(input, input_reversed, output, model, reverse_model):
  output_reversed = output.to(device_r)
  inputs = torch.cat((input, output[:, :-1]), dim=1)
  inputs_reversed = torch.cat((output_reversed, input_reversed[:, :-1]), dim=1)
  mask = torch.full_like(input[:, :-1], -1, dtype=torch.long)
  labels = torch.cat((mask, output), dim=1)
  mask_reversed = torch.full_like(
      output_reversed[:, :-1], -1, dtype=torch.long)
  labels_reversed = torch.cat((mask_reversed, input_reversed), dim=1)

  loss, *_ = model(inputs, labels=labels)
  reverse_loss, *_ = reverse_model(inputs_reversed, labels=labels_reversed)

  return -(ALPHA * loss.float() + (1 - ALPHA) * reverse_loss.float())


def append_messages(old_list: list, new_list: list, vocab, end_token,
                    truncate_length=64):
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


def decode(ids, vocab, skip_special_tokens=True):
  gen = vocab.to_tokens(ids)
  sent = ''
  for word in gen:
    word = word.replace('▁', ' ')
    word = word.replace(' !', '!')
    word = word.replace(' .', '.')
    word = word.replace(' ?', '?')
    word = word.replace(' ,', ',')
    if skip_special_tokens:
      word = word.replace('<unk>', '')
      word = word.replace('<s>', '')
      word = word.replace('</s>', '')
    sent += word
  return sent[1:]


def generate_message(message_list: list, model, reverse_model, vocab,
                     focus_last_message=True):
  total_input = torch.cat(message_list, dim=1).to(device_f)
  if focus_last_message:
    total_input_reversed = message_list[-1]
  else:
    total_input_reversed = torch.cat(list(reversed(message_list)), dim=1)

  outputs = model.generate(input_ids=total_input,
                           min_length=total_input.shape[1] + 8,
                           max_length=total_input.shape[1] + 40,
                           num_return_sequences=num_samples,
                           top_k=top_k,
                           top_p=top_p,
                           do_sample=True,
                           repetition_penalty=1.2,
                           pad_token_id=PADDING_TOKEN,
                           eos_token_id=vocab[vocab.eos_token])
  outputs = outputs[:, total_input.shape[1]:]

  scores = []
  for output in outputs:
    output = output.unsqueeze(0).to(device_f)
    try:
      output = output[:, :output[0].tolist().index(vocab[vocab.eos_token]) + 1]
    except:
      pass
    scores.append(_score_response(
        total_input, total_input_reversed.to(device_r), output,
        model, reverse_model))
  scores = torch.stack(scores, dim=0)

  # import pdb
  # pdb.set_trace()

  # winner = torch.multinomial(
  #     F.softmax(scores / MMI_temperature, dim=0), num_samples=1).item()
  winner = torch.argmax(scores).item()
  out = outputs[winner]

  return decode(out.tolist(), vocab)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Interact with the model.')
  parser.add_argument('vocab_path', metavar='vocab_path', type=str,
                      help='Vocabulary path')
  parser.add_argument('model_path', metavar='model_path',
                      type=str, help='Model path')
  parser.add_argument('reverse_model_path', metavar='reverse_model_path',
                      type=str, help='Reverse model path')

  args = parser.parse_args()

  # 사전, 모델 파일 및 end_token을 불러옵니다.
  vocab, model, reverse_model, end_token = load(
      args.vocab_path, args.model_path, args.reverse_model_path)

  my_message_list = []  # 대화 히스토리 리스트
  while True:
    my_message = input('usr >> ')
    # 대화 히스토리에 사용자가 입력한 내용을 추가합니다.
    append_messages(my_message_list, [my_message], vocab, end_token)

    # 대화 히스토리를 바탕으로 답변을 생성합니다.
    my_response = generate_message(
        my_message_list, model, reverse_model, vocab, False)
    print('bot >>', my_response)

    # 답변을 대화 히스토리에 추가합니다.
    append_messages(my_message_list, [my_response], vocab, end_token)
