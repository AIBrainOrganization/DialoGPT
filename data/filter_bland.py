from tqdm import tqdm

import pandas as pd

import json

import sys
sys.path.append('/home/calee/git/dcinside')
from filter import text_to_words
from filter import Comment
from filter import filter_rows

TRIGRAMS_PATH = 'trigrams'
FILE_PATHS = ['train_bland.tsv', 'valid_bland.tsv', 'test_bland.tsv']


def get_trigrams(s):
  s = text_to_words(s)
  return [s[i:i + 3] for i in range(len(s) - 2)]


def read_or_create_trigrams():
  try:
    with open(TRIGRAMS_PATH) as f:
      trigrams = json.loads(f.read())
  except IOError:
    trigrams = {}
    for file in FILE_PATHS:
      df = pd.read_csv(file, sep='\t', header=None)
      for _, line in tqdm(df.iterrows()):
        source, target = line
        source = source.split(' EOS ')
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

    with open(TRIGRAMS_PATH, 'w') as f:
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


def create_rows(trigrams):
  ret = []
  for file in FILE_PATHS:
    rows = []
    df = pd.read_csv(file, sep='\t', header=None)
    for _, line in tqdm(df.iterrows()):
      source, target = line
      sentences = source.split(' EOS ')
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
  trigrams = read_or_create_trigrams()
  rowss = create_rows(trigrams)
  filtered = [filter_rows(r) for r in rowss]

  df = [pd.DataFrame(f) for f in filtered]
  names = ['train.tsv', 'valid.tsv', 'test.tsv']
  for i, d in enumerate(df):
    d.to_csv(names[i], sep='\t', header=False, index=False)


if __name__ == '__main__':
  main()
