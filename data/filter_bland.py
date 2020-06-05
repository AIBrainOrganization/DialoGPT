from tqdm import tqdm

import pandas as pd

import json
import argparse
import os

import sys
sys.path.append('/home/calee/git/dcinside')
from filter import text_to_words
from filter import Comment
from filter import filter_rows

TRIGRAMS_PATH = 'trigrams'


def get_trigrams(s):
  s = text_to_words(s)
  return [s[i:i + 3] for i in range(len(s) - 2)]


def read_or_create_trigrams(args):
  try:
    with open(os.path.join(args.path, TRIGRAMS_PATH)) as f:
      trigrams = json.loads(f.read())
  except IOError:
    trigrams = {}
    for file in args.files:
      df = pd.read_csv(os.path.join(args.path, file), sep='\t', header=None)
      for _, line in tqdm(df.iterrows()):
        source, target, next_state = line
        source = source.split(' EOS ')
        sentences = source
        sentences.append(target)
        sentences.append(next_state)
        sentences = [s[4:] for s in sentences if int(s[0])]
        sentences = [[' '.join(t) for t in get_trigrams(s)] for s in sentences]

        for s in sentences:
          for t in s:
            if t in trigrams:
              trigrams[t] += 1
            else:
              trigrams[t] = 1

    with open(os.path.join(args.path, TRIGRAMS_PATH), 'w') as f:
      f.write(json.dumps(trigrams))

  s = sorted(trigrams.items(), key=lambda k: -k[1])
  global BLAND_THRESHOLD
  BLAND_THRESHOLD = s[BLAND_TOP][1]

  return trigrams


BLAND_TOP = 10000
BLAND_THRESHOLD = None


def check_bland(s, trigrams):
  ts = [' '.join(t) for t in get_trigrams(s)]
  n = len(ts)
  if n == 0:
    return False
  threshold = 0.9 * n
  for t in ts:
    if trigrams[t] <= BLAND_THRESHOLD:
      n -= 1
      if n < threshold:
        return False
  return True


def create_rows(trigrams, args):
  ret = []
  for file in args.files:
    rows = []
    df = pd.read_csv(os.path.join(args.path, file), sep='\t', header=None)
    for _, line in tqdm(df.iterrows(), total=df.shape[0]):
      source, target, next_state = line
      sentences = source.split(' EOS ')
      sentences.append(target)
      sentences.append(next_state)

      sentences = [Comment(s[4:], [] if int(s[0]) else ['bad'])
                   for s in sentences]

      for s in sentences:
        if len(s.tags) == 0 and check_bland(s.comment, trigrams):
          s.tags.append('bland')
      rows.append(sentences)
    ret.append(rows)
  return ret


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str,
                      help='directory of the files')
  parser.add_argument("--files", type=str, nargs='+',
                      help='name of the files')
  args = parser.parse_args()

  trigrams = read_or_create_trigrams(args)
  rowss = create_rows(trigrams, args)
  filtered = [filter_rows(r) for r in rowss]

  df = [pd.DataFrame(f) for f in filtered]
  names = ['train_reinforce.tsv', 'valid_reinforce.tsv', 'test_reinforce.tsv']
  for i, d in enumerate(df):
    d.to_csv(os.path.join(args.path, names[i]),
             sep='\t', header=False, index=False)


if __name__ == '__main__':
  main()
