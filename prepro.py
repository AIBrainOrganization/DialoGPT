#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
preprocess input data into feature and stores binary as python shelve DB
each chunk is gzipped JSON string
bos_token: 0
eos_token: 1
padding_token: 3 ?? 0
"""
import argparse
import gzip
import json
import sys
import shelve
import os
import torch
import subprocess as sp
import pandas as pd

from os.path import dirname, exists, join
from tqdm import tqdm
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from kogpt2.utils import get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
from gpt2_training.train_utils import InputFeatures_train as InputFeatures


def _get_file_len(corpus):
  n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                               universal_newlines=True).split()[0])
  return n_line


def _get_inputs_from_text(text, tokenizer, vocab):
  input, reward = text
  input = vocab[tokenizer(input)]

  reward = float(reward)

  return input, reward


def _make_features(id_, input, reward, vocab, max_len):
  end_of_text_id = vocab[vocab.eos_token]
  features = []

  if len(input) > max_len:
    sys.exit('max_len exceeded.')
  feat = _make_feature(id_, input, reward, end_of_text_id)
  if feat is not None:
    features.append(feat)

  return features


def _make_feature(id_, sent, reward, eos):
  if len(sent) == 0:
    import pdb
    pdb.set_trace()
  return InputFeatures(id_, sent, reward)


def main(args):
  # toker = GPT2Tokenizer.from_pretrained('gpt2')
  tok_path = get_tokenizer()
  toker = SentencepieceTokenizer(tok_path)
  _, vocab = get_pytorch_kogpt2_model()
  attrs = []
  if args.two_turn:
    attrs.append('2turn')
  if attrs:
    db_path = (f'{args.corpus[:-4]}.{args.max_seq_len}len.'
               f'{".".join(attrs)}.db/db')
  else:
    db_path = f'{args.corpus[:-4]}.{args.max_seq_len}len.db/db'
  if exists(dirname(db_path)):
    raise ValueError('Found existing DB, please backup')
  else:
    os.makedirs(dirname(db_path))
  with shelve.open(db_path, 'n') as db:
    # reader = open(args.corpus, "r", encoding="utf-8")
    reader = pd.read_csv(args.corpus, sep='\t', header=None)
    chunk = []
    n_chunk = 0
    n_example = 0

    # print("pdb-attach")
    # from pdb_clone import pdb
    # rsock = pdb.set_trace_remote()
    #
    # if rsock.state != rsock.ST_CONNECTED:
    #   input()

    for _, line in tqdm(reader.iterrows(), total=len(reader)):
      try:
        if len(chunk) >= args.chunk_size:
          # save and renew chunk
          db[f'chunk_{n_chunk}'] = gzip.compress(
              json.dumps(chunk[:args.chunk_size]).encode('utf-8'))
          chunk = chunk[args.chunk_size:]
          n_chunk += 1

        input, reward = _get_inputs_from_text(
            line, toker, vocab)
        features = _make_features(
            n_example, input, reward, vocab, args.max_seq_len)
        for feature in features:
          chunk.append(vars(feature))
          n_example += 1
      except Exception as e:
        print('!!! prepro exception !!!', e)
        continue
    # save last chunk
    db[f'chunk_{n_chunk}'] = gzip.compress(
        json.dumps(chunk).encode('utf-8'))
  # save relevant information to reproduce
  meta = {'n_example': n_example,
          'chunk_size': args.chunk_size,
          'max_seq_len': args.max_seq_len,
          'two_turn': args.two_turn}
  with open(join(dirname(db_path), 'meta.json'), 'w') as writer:
    json.dump(meta, writer, indent=4)
  # torch.save(toker, join(dirname(db_path), 'tokenizer.pt'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--corpus', required=True,
                      help='file name of training corpus (should be .tsv)')
  parser.add_argument('--chunk_size', type=int, default=65536,
                      help='num of data examples in a storing chunk')
  parser.add_argument('--max_seq_len', type=int, default=128,
                      help='discard data longer than this')
  parser.add_argument('--two_turn', action='store_true',
                      help='take only the last 2 turns')

  args = parser.parse_args()

  main(args)
