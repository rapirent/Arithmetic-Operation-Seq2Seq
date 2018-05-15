# coding: utf-8
from keras.models import Sequential
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from six.moves import range

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_size', default='45000')
parser.add_argument('--train_size', default='40000')
parser.add_argument('--digits', default='3')
parser.add_argument('--epoch', default='2')
parser.add_argument('--activation', default='softmax')
parser.add_argument('--output_name', default='model_1')
args = parser.parse_args()

# # Parameters Config
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

DATA_SIZE = int(args.data_size)
TRAIN_SIZE = int(args.train_size)
DIGITS = int(args.digits)
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+-* '
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
EPOCH_SIZE = int(args.epoch)
LAYERS = 1
ACTIVATION = args.activation

output_file = open('./data/as-' + args.output_name, 'w')
print('DATA_SIZE = ', DATA_SIZE , file=output_file)
print('TRAIN_SIZE = ', TRAIN_SIZE, file=output_file)
print('DIGITS = ', DIGITS, file=output_file)
print('EPOCH_SIZE = ', EPOCH_SIZE, file=output_file)
print('ACTIVATION = ', ACTIVATION, file=output_file)

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


# # Data Generation

questions = []
expected = []
seen = set()
print('Generating data...')

while len(questions) < DATA_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    if len(questions) % 3 == 1:
        q = '{}-{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a - b)
    elif len(questions) % 3 == 2:
        q = '{}+{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a + b)
    else:
        q = '{}*{}'.format(a, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a * b)
    if q in seen:
        continue
    seen.add(q)
    ans += ' ' * (MAXLEN - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

print(questions[:5], expected[:5])


# # Processing
print('Vectorization... (to the one-hot encoding)')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), MAXLEN, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, MAXLEN)

indices = np.arange(len(y))
np.random.shuffle(indices)
print(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = x[:TRAIN_SIZE]
train_y = y[:TRAIN_SIZE]
test_x = x[TRAIN_SIZE:]
test_y = y[TRAIN_SIZE:]

print('Training Data:')
print(train_x.shape)
print(train_y.shape)

split_at = len(train_x) - len(train_x) // 10
print('split_at', split_at)
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)


# In[12]:


print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])


# # Build Model

print('Build model...')
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(MAXLEN))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(chars))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


print('train set = ', x_train.shape, 'validation set = ', x_val.shape, file=output_file)
acc = []
val_acc = []
loss = []
val_loss = []
# # Training
for loop in range(100):
    print()
    print('-' * 50)
    print('Train Loop Num:', loop)
    history = model.fit(x_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCH_SIZE,
                validation_data=(x_val, y_val),
                shuffle=True)
    acc += history.history['acc']
    val_acc += history.history['val_acc']
    loss += history.history['loss']
    val_loss += history.history['val_loss']
    print('loop ', loop, file=output_file)
    print('acc = {}  '.format(history.history['acc']), end='', file=output_file)
    print('val_acc = {}  '.format(history.history['val_acc']), end='', file=output_file)
    print('loss = {}  '.format(history.history['loss']), end='', file=output_file)
    print('val_loss = {}  '.format(history.history['val_loss']), file=output_file)
    print('-' * 50 , file=output_file)

    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)


# # Testing
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
print("MSG : Accuracy is {}".format(right / len(preds)), file=output_file)
model.save('./models/m-' + args.output_name + '.h5')
with open('./corpus/m-' + args.output_name + '-training-corpus.csv', 'w') as corpus:
    print('questions,expected', file=corpus)
    for (x, y) in zip(x_train, y_train):
        print('{},{}'.format(ctable.decode(x), ctable.decode(y)), file=corpus)
with open('./corpus/m-' + args.output_name + '-validation-corpus.csv', 'w') as corpus:
    print('questions,expected', file=corpus)
    for (x, y) in zip(x_val, y_val):
        print('{},{}'.format(ctable.decode(x), ctable.decode(y)), file=corpus)
with open('./corpus/m-' + args.output_name + '-testing-corpus.csv', 'w') as corpus:
    print('questions,expected', file=corpus)
    for (x, y) in zip(test_x, test_y):
        print('{},{}'.format(ctable.decode(x), ctable.decode(y)), file=corpus)
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./fig/m-accuracy-' + args.output_name + '.png')
plt.clf()
# summarize history for loss
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./fig/m-loss-' + args.output_name + '.png')
output_file.close()
plt.clf()
