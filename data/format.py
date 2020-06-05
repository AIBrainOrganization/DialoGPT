import argparse
import re
import os


def format_line(line):
  line = re.sub(r'\\+"([^\n])', '""\\1', line)
  line = re.sub(r'\\+\'', '\'', line)
  line = re.sub(r'\\', '', line)

  return line


def main():
  parser = argparse.ArgumentParser(
      description='Format csv files for pandas')
  parser.add_argument('input', metavar='input', type=str,
                      help='Input path')
  parser.add_argument('output', metavar='output', type=str, help='Save path')

  args = parser.parse_args()

  train = open(os.path.join(args.output, 'train_bland.tsv'), 'w')
  valid = open(os.path.join(args.output, 'valid_bland.tsv'), 'w')
  test = open(os.path.join(args.output, 'test_bland.tsv'), 'w')

  with open(args.input) as input:
    for i, line in enumerate(input):
      line = format_line(line)
      output = train
      if i % 10 == 0:
        output = valid
      elif i % 10 == 1:
        output = test
      output.write(line)

  train.close()
  valid.close()
  test.close()


if __name__ == "__main__":
  main()
