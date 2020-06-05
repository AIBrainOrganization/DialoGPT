from tqdm import tqdm

import pandas as pd

import json
import argparse
import os
import sys

sys.path.append('/home/calee/git/text_emotion_recognition')
from sentiment_analysis import sentiment_analysis as emotion


HAPPINESS_REWARD = 1
ANGER_REWARD = -1
DISGUST_REWARD = -1
FEAR_REWARD = -1
NEUTRAL_REWARD = 0
SADNESS_REWARD = -1
SURPRISE_REWARD = 0


def get_rewards(sentences, batch_size=32):
  ret = []
  for i in tqdm(range(0, len(sentences), batch_size)):
    ess = json.loads(emotion(sentences[i:i + batch_size]))
    for es in ess:
      for e in es['predict_emotion']:
        description = e['description']
        ratio = float(e['percentage'][:-1]) / 100
        if description == 'HAPPINESS':
          happiness = ratio
        elif description == 'ANGER':
          anger = ratio
        elif description == 'DISGUST':
          disgust = ratio
        elif description == 'FEAR':
          fear = ratio
        elif description == 'NEUTRAL':
          neutral = ratio
        elif description == 'SADNESS':
          sadness = ratio
        elif description == 'SURPRISE':
          surprise = ratio

      ret.append(HAPPINESS_REWARD * happiness + ANGER_REWARD * anger +
                 DISGUST_REWARD * disgust + FEAR_REWARD * fear +
                 NEUTRAL_REWARD * neutral + SADNESS_REWARD * sadness +
                 SURPRISE_REWARD * surprise)
  return ret


def attach_reward(args):
  ret = []
  for file in args.files:
    df = pd.read_csv(os.path.join(args.path, file), sep='\t', header=None)
    rewards = pd.DataFrame(get_rewards(df[2]))
    df = pd.concat([df, rewards], axis=1, ignore_index=True)
    ret.append(df)
  return ret


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str,
                      help='directory of the files')
  parser.add_argument("--files", type=str, nargs='+',
                      help='name of the files')
  args = parser.parse_args()

  df = attach_reward(args)
  names = ['train.tsv', 'valid.tsv', 'test.tsv']
  for i, d in enumerate(df):
    d.to_csv(os.path.join(args.path, names[i]),
             sep='\t', header=False, index=False)


if __name__ == '__main__':
  main()
