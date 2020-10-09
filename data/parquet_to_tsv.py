from tqdm import tqdm

import pandas as pd

import argparse
import json
import os

from dcinside import filter as dcinside_filter
from dcinside.filter import Comment
from dcinside.filter import filter_rows
from dcinside.filter import add_tags
from dcinside.clean import clean


def create_rows(input_path):
  train = []
  valid = []
  test = []

  df = pd.read_parquet(input_path)
  for index, row in tqdm(df.iterrows(), total=len(df.index), desc='create_rows'):
    try:
      doc = json.loads(row.contents)
      cleaned = clean(doc)

      handle = train
      if index % 10 == 0:
        handle = test
      elif index % 10 == 1:
        handle = valid
      handle += dcinside_filter.create_rows(cleaned)
    except TypeError:
      pass

  return train, valid, test


def main():
  parser = argparse.ArgumentParser(description='Parquet files to tsv.')
  parser.add_argument('input', metavar='input', type=str, help='Input path')
  parser.add_argument('output', metavar='output', type=str, help='Output path')

  args = parser.parse_args()

  rows = create_rows(args.input)

  filtered = [filter_rows(r) for r in rows]

  df = [pd.DataFrame(f) for f in filtered]
  names = ['train_bland.tsv', 'valid_bland.tsv', 'test_bland.tsv']
  for i, d in enumerate(df):
    d.to_csv(os.path.join(args.output, names[i]), sep='\t', header=False, index=False)


if __name__ == '__main__':
  main()
