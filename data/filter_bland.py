from tqdm import tqdm

import pandas as pd

import json
import argparse
import os
import re

from dcinside.filter import text_to_words
from dcinside.filter import Comment
from dcinside.filter import filter_rows

TRIGRAMS_PATH = 'trigrams'
FILE_PATHS = ['train_bland.tsv', 'valid_bland.tsv', 'test_bland.tsv']
SPLIT_PATTERN = re.compile(r' EOS (?=[01]\.0)')


def get_trigrams(s):
  s = text_to_words(s)
  return [s[i:i + 3] for i in range(len(s) - 2)]


def split(source):
  return SPLIT_PATTERN.split(source)


def read_or_create_trigrams(path):
  try:
    with open(os.path.join(path, TRIGRAMS_PATH)) as f:
      trigrams = json.load(f)
  except IOError:
    trigrams = {}
    for file in FILE_PATHS:
      df = pd.read_csv(os.path.join(path, file), sep='\t', header=None)
      for _, line in tqdm(df.iterrows(), total=len(df.index), desc='create trigrams'):
        source, target = line
        source = split(source)
        sentences = source
        sentences.append(target)
        sentences = [s[4:] for s in sentences if int(s[0])]

        sentences = [[' '.join(t) for t in get_trigrams(s)] for s in sentences]

        for s in sentences:
          for t in s:
            if t in trigrams:
              trigrams[t] += 1
            else:
              trigrams[t] = 1

    with open(os.path.join(path, TRIGRAMS_PATH), 'w') as f:
      f.write(json.dumps(trigrams))

  s = sorted(trigrams.items(), key=lambda k: -k[1])
  global BLAND_THRESHOLD
  BLAND_THRESHOLD = s[BLAND_TOP][1]

  return trigrams


BLAND_TOP = 10000
BLAND_THRESHOLD = None


def check_bland(s, trigrams):
  ts = [' '.join(t) for t in get_trigrams(s)]
  n = 0
  for t in ts:
    if trigrams[t] > BLAND_THRESHOLD:
      n += 1

  try:
    return n / len(ts) >= 0.9
  except ZeroDivisionError:
    return False


def create_rows(trigrams, path):
  ret = []
  for file in FILE_PATHS:
    rows = []
    df = pd.read_csv(os.path.join(path, file), sep='\t', header=None)
    for _, line in tqdm(df.iterrows(), total=len(df.index), desc='create rows'):
      source, target = line
      sentences = split(source)
      sentences.append(target)

      sentences = [Comment(s[4:], [] if int(s[0]) else ['bad'])
                   for s in sentences]

      for s in sentences:
        if len(s.tags) == 0 and check_bland(s.comment, trigrams):
          if len(text_to_words(s.comment)) > 3:
            print(s.comment)
          s.tags.append('bland')

      rows.append(sentences)
    ret.append(rows)
  return ret


def main():
  parser = argparse.ArgumentParser(description='Filter bland phrases.')
  parser.add_argument('path', metavar='path', type=str, help='Directory path')

  args = parser.parse_args()

  trigrams = read_or_create_trigrams(args.path)
  rowss = create_rows(trigrams, args.path)
  filtered = [filter_rows(r) for r in rowss]

  df = [pd.DataFrame(f) for f in filtered]
  names = ['train.tsv', 'valid.tsv', 'test.tsv']
  for i, d in enumerate(df):
    d.to_csv(os.path.join(args.path, names[i]), sep='\t', header=False, index=False)


if __name__ == '__main__':
  main()
