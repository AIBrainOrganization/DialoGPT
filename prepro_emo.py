#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
"""
Change a tsv file to db format.

preprocess input data into feature and stores binary as python shelve DB
each chunk is gzipped JSON string
bos_token: 0
eos_token: 1
padding_token: 3 ?? 0
"""
import argparse
import gzip
import json
import subprocess as sp
import shelve
import os
from os.path import dirname, exists, join

import torch
# from lsp_model import GPT2Tokenizer
from tqdm import tqdm

# from env import END_OF_TEXT_TOKEN
from gpt2_training.train_utils import InputFeatures_train as InputFeatures

from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer

## get tokenizer from koGPT2
from kogpt2.utils import get_tokenizer

import pandas as pd


## tsv 파일이 총 몇 줄인지
def _get_file_len(corpus):
    n_line = int(sp.check_output(f"wc -l {corpus}".split(),
                                 universal_newlines=True).split()[0])
    return n_line


## text 받아서 w에 weight 할당
def _norm_text(text):
    w, *toks, e = text.strip().split()  # weight와 token으로 분리

    try:
        # w = weight 0.0 or 1.0
        w = float(w)
    except Exception:
        toks = [w] + toks
        w = 1.0
    return w, ' '.join(toks), e  # 단어 token을 합쳐 하나의 문장으로 반환


## text를 입력으로 받아 해당 문장의 weights와 tokenizer 결과를 list로 반환하는 함수
def _get_inputs_from_text(text, tokenizer, vocab):
    srcs, tgt = text  # text는 source, target 문장으로 나누어져 있음
    weights = []
    inputs = []
    emotions = []
    for src in srcs.split(' EOS '):  # 문장별 잘라서 사용
        src_weight, src, src_emotion = _norm_text(src)
        context_id = vocab[tokenizer(src)]
        weights.append(src_weight)
        inputs.append(context_id)
        emotions.append(vocab[src_emotion])
    tgt_weight, tgt, tgt_emotion= _norm_text(tgt)
    if tgt_weight != 0:  # wieght == 1인 경우만 학습 데이터에 사용
        response_id = vocab[tokenizer(tgt)]
        weights.append(tgt_weight)
        inputs.append(response_id)
        emotions.append(vocab[tgt_emotion])
    return weights, inputs, emotions


def _make_features(id_, weights, inputs, emotions, tokenizer, vocab, max_len):
    end_of_text_id = vocab[vocab.eos_token]
    features = []
    sents = []
    ws = []
    es = []
    len_ = 0
    i = 0

    for ids, w, e in zip(inputs, weights, emotions):
        if len(ids) > max_len:  # 현재 문장의 토큰 길이가 max_len보다 길다면
            if len(sents) >= 2:  # 문장들을 담는 sents list의 길이가 2보다 커지면
                feat = _make_feature(id_ + i, sents, ws, es, end_of_text_id)
                if feat is not None:
                    features.append(feat)
                    i += 1
            len_ = 0
            sents = []  # # 길이가 너무 긴 문장이 들어왔을 경우 sents의 길이가 1이면 긴 문장과 sents에 있는 문장 토큰을 버림
            ws = []  # ws 초기화
            es = [] # es 초기화
            continue
        elif len_ > max_len:  # 여러 문장 토큰의 합이 max_len보다 크면
            feat = _make_feature(id_ + i, sents, ws, end_of_text_id)
            if feat is not None:
                features.append(feat)
                i += 1
            len_ = len(sents[-1]) + 1 + 1 # 감정 토큰 길이가 추가되므로
            sents = sents[-1:]
            ws = ws[-1:]
            es = es[-1:]
        len_ += (len(ids) + 1 + 1)  # 감정 토큰이 추가되므로 길이 (emo) + (eos) 로 추가
        sents.append(ids)
        ws.append(w)
        es.append(e)
    if len(sents) >= 2:
        feat = _make_feature(id_ + i, sents, ws, es, end_of_text_id)
        if feat is not None:
            features.append(feat)

    return features


def _make_feature(id_, sents, ws, es, eos):
    if all(w == 0 for w in ws[1:]):  # 입력으로 들어온 sents의 weights가 모두 0이면 None을 반환
        return None
    input_ids = [i for s, e in zip(sents, es) for i in s + [e, eos]][:-1]  # 문장이 끝나는 지점에 1(eos) 추가
    lm_labels = []
    weights = []
    emotion_ids = []
    token_type_ids = []  # this becomes round ids
    for i, (s, w, e) in enumerate(zip(sents, ws, es)):  # 한 문장의 token을 사용하면서
        if i == 0:  # 첫번째 문장이면  # emotion token까지 고려하여 계산 필요
            lm_labels += [-1] * (len(s) + 1)
            weights += [0.0] * (len(s) + 1)
            token_type_ids += [0] * (len(s) + 1)
            emotion_ids += [e] * (len(s) + 1)
            continue

        token_type_ids += [i] * (len(s) + 2)
        emotion_ids +=  [e] * (len(s) + 2)
        if w == 0.0:
            lm_labels += [-1] * (len(s) + 2)
            weights += [0.0] * (len(s) + 2)
        else:
            lm_labels += (s + [e, eos])
            weights += [w] * (len(s) + 2)

    # handle trailing -1's
    i = len(lm_labels) - 1
    while i >= 0:
        if lm_labels[i] != -1:
            break
        i -= 1
    input_ids = input_ids[:i + 1]
    lm_labels = lm_labels[:i + 1]
    weights = weights[:i + 1]
    token_type_ids = token_type_ids[:i + 1]
    emotion_ids = emotion_ids[:i+1]

    # pad to multiples of 8
    while len(input_ids) % 8 != 0:
        input_ids.append(0)
        token_type_ids.append(0)
        lm_labels.append(-1)
        weights.append(0.0)
        emotion_ids.append(0)

    position_ids = list(range(len(input_ids)))
    assert (len(input_ids) == len(position_ids) == len(token_type_ids)
            == len(lm_labels) == len(weights) == len(emotion_ids))
    assert len(input_ids) % 8 == 0
    if len(input_ids) == 0:
        import pdb
        pdb.set_trace()
    feature = InputFeatures(id_, input_ids, position_ids, token_type_ids,
                            lm_labels, weights, emotion_ids)
    return feature


def main(args):
    # toker = GPT2Tokenizer.from_pretrained('gpt2')
    tok_path = get_tokenizer()  # koGPT2 tokenizer
    toker = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)  # gluonnlp
    _, vocab = get_pytorch_kogpt2_model()
    '''
    def get_pytorch_kogpt2_model(ctx='cpu', cachedir='~/kogpt2/'):
    # download model
    model_info = pytorch_kogpt2
    model_path = _download(model_info['url'],
                           model_info['fname'],
                           model_info['chksum'],
                           cachedir=cachedir)
    # download vocab
    vocab_info = tokenizer
    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)
    return get_kogpt2_model(model_path, vocab_path, ctx)
    '''

    attrs = []
    if args.reverse:
        attrs.append('reverse')
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

    ## shelve.open : open dict
    with shelve.open(db_path, 'n') as db:
        # reader = open(args.corpus, "r", encoding="utf-8")
        reader = pd.read_csv(args.corpus, sep='\t', header=None)  # read .tsv file
        chunk = []
        n_chunk = 0
        n_example = 0

        # print("pdb-attach")
        # from pdb_clone import pdb
        # rsock = pdb.set_trace_remote()
        #
        # if rsock.state != rsock.ST_CONNECTED:
        #   input()

        for _, line in tqdm(reader.iterrows()):
            try:
                if len(chunk) >= args.chunk_size:  # ?
                    # save and renew chunk
                    db[f'chunk_{n_chunk}'] = gzip.compress(  # ?
                        json.dumps(chunk[:args.chunk_size]).encode('utf-8'))
                    chunk = chunk[args.chunk_size:]
                    n_chunk += 1

                weights, inputs, emotions = _get_inputs_from_text(line, toker, vocab)
                if args.reverse:
                    weights = list(reversed(weights))
                    inputs = list(reversed(inputs))
                    emotions = list(reversed(emotions))
                if args.two_turn:
                    weights = weights[:2]
                    inputs = inputs[:2]
                    emotions = emotions[:2]
                if len(weights) < 2:
                    continue
                features = _make_features(n_example, weights, inputs, emotions,
                                          toker, vocab, args.max_seq_len)

                '''
                def _make_features(id_, weights, inputs, tokenizer, vocab, max_len):
                  end_of_text_id = vocab[vocab.eos_token]
                  features = []
                  sents = []
                  ws = []
                  len_ = 0
                  i = 0
        
                  return features
                '''
                for feature in features:
                    chunk.append(vars(feature))  # vars = dict/object 그대로 넘김
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
            'reverse': args.reverse,
            'two_turn': args.two_turn}
    with open(join(dirname(db_path), 'meta.json'), 'w') as writer:
        json.dump(meta, writer, indent=4)
    # torch.save(toker, join(dirname(db_path), 'tokenizer.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=False, default="./data/5Y_emotion_valid.tsv",
                        help='file name of training corpus (should be .tsv)')
    parser.add_argument('--chunk_size', type=int, default=65536,
                        help='num of data examples in a storing chunk')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help='discard data longer than this')
    parser.add_argument('--reverse', action='store_true',
                        help='reverse the src tgt')
    parser.add_argument('--two_turn', action='store_true',
                        help='take only the first 2 turns')

    args = parser.parse_args()

    main(args)
