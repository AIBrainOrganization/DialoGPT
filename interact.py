import torch
import argparse

import torch.nn.functional as F

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from config import device_f, device_r, num_samples
from config import top_k, top_p, ALPHA, BETA
from kogpt2.pytorch_kogpt2 import get_kogpt2_model
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from util import get_device

PADDING_TOKEN = 0

torch.set_grad_enabled(False)

# 토크나이저를 불러옵니다.
tok_path = get_tokenizer()
tokenizer = SentencepieceTokenizer(tok_path)


# 사전, 모델, 리버스 모델을 불러옵니다.
def load(vocab_path, model_path, reverse_path, device=0):
  # 모델과 사전을 불러옵니다.
  model, vocab = get_kogpt2_model(model_path, vocab_path, device_f)

  if device_f == 'cuda' or type(device_f) == int and device_f >= 0:
    model.half()
  model.eval()

  # 리버스 모델을 불러옵니다.
  reverse_model, _ = get_kogpt2_model(reverse_path, vocab_path, device_r)
  if device_r == 'cuda' or type(device_r) == int and device_r >= 0:
    reverse_model.half()
  reverse_model.eval()

  end_token = torch.tensor([[vocab[vocab.eos_token]]], dtype=torch.long)

  return vocab, model, reverse_model, end_token


# 각 답변의 점수를 계산합니다.
def _score_response(input, input_reversed, output, model, reverse_model, dqn=None):
  # input, label, mask를 준비합니다.
  device_r = get_device(reverse_model)
  output_reversed = output.to(device_r)
  inputs = torch.cat((input, output[:, :-1]), dim=1)
  inputs_reversed = torch.cat((output_reversed, input_reversed[:, :-1]), dim=1)
  mask = torch.full_like(input[:, :-1], -1, dtype=torch.long)
  labels = torch.cat((mask, output), dim=1)
  mask_reversed = torch.full_like(
      output_reversed[:, :-1], -1, dtype=torch.long)
  labels_reversed = torch.cat((mask_reversed, input_reversed), dim=1)

  # 점수로 활용될 loss 값을 계산합니다.
  loss, *_ = model(inputs, labels=labels)
  reverse_loss, *_ = reverse_model(inputs_reversed, labels=labels_reversed)
  if dqn is None:
    # ALPHA 값으로 비중을 주고 loss를 점수로 변경하기 위해 -1을 곱해줍니다.
    return -(ALPHA * loss + (1 - ALPHA) * reverse_loss)
  else:
    inputs = inputs.to(get_device(dqn))
    value = dqn(inputs)
    return (BETA - 1) * \
        (ALPHA * loss + (1 - ALPHA) * reverse_loss) + \
        BETA * value


# 히스토리에 새 문장을 추가해줍니다.
def append_messages(old_list: list, new_list: list, vocab, end_token,
                    truncate_length=64):
  for message in new_list:
    if message != '':
      # 문장을 tokenizing합니다.
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


# 생성된 token들을 문장으로 바꿔줍니다.
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


# 답변 문장을 생성합니다.
def generate_message(message_list: list, model, reverse_model, vocab,
                     focus_last_message=True):
  total_input = torch.cat(message_list, dim=1).to(device_f)
  if focus_last_message:
    total_input_reversed = message_list[-1]
  else:
    total_input_reversed = torch.cat(list(reversed(message_list)), dim=1)

  # https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.PreTrainedModel.generate
  # 후보 답변 문장들을 생성합니다.
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

  # 각 문장에 대해 점수를 계산합니다.
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

  # 가장 점수가 높은 문장을 선택합니다.
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
