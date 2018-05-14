# coding: utf-8
from keras.models import Sequential, load_model
from keras.models import Sequential
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from six.moves import range

import os
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='model_1')
parser.add_argument('--digits', default='3')
args = parser.parse_args()
chars = '0123456789+-* '

DIGITS = int(args.digits)
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)

ctable = CharacterTable(chars)
ctable.indices_char

model = load_model('./models/m-' + args.model_name + '.h5')
test_x = []
test_y = []
corpus = open('./corpus/m-' + args.model_name + '-testing-corpus.csv', 'r')
corpus_reader = csv.DictReader(corpus)
corpus_len = sum(1 for row in corpus_reader)
corpus.close()
corpus = open('./corpus/m-' + args.model_name + '-testing-corpus.csv', 'r')
corpus_reader = csv.DictReader(corpus)
# load corpus
test_x = np.zeros((corpus_len, MAXLEN, len(chars)), dtype=np.bool)
test_y = np.zeros((corpus_len, MAXLEN, len(chars)), dtype=np.bool)
for i, row in enumerate(corpus_reader):
    test_x[i] = ctable.encode(row['questions'], MAXLEN)
    test_y[i] = ctable.encode(row['expected'], MAXLEN)

print("MSG : Prediction")
print("-" * 50)
right = 0
preds = model.predict_classes(test_x, verbose=0)
for i in range(len(preds)):
    q = ctable.decode(test_x[i])
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(preds[i], calc_argmax=False)
    print('Q', q[::-1] if REVERSE else q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
        right += 1
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)
print("MSG : Accuracy is {}".format(right / len(preds)))
