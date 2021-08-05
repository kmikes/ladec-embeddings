# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""

from __future__ import division, print_function, absolute_import

import pandas as pd
import numpy as np

from numpy import random

import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

# Load Data
data = pd.read_csv('NoteBooks/data/all_embeddings_forML.csv')

voc = np.unique(data[['c1', 'c2', 'cmp']].values.reshape(-1))
# print( voc.shape )

dims = 50

# Shuffle the data
random.seed(1)
np.random.seed(1)

df = data.copy()
# print(df.head())
df = df.sample(frac=1).reset_index(drop=True)
# print(df.head())

# EMBEDDINGS DICT
from scipy import spatial

embeddings_dict = {}

with open("NoteBooks/data/glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector


def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


# SETUP VISIBLE TESTING
compounds = list(df['cmp'])
rows = len(compounds)
rows_train = int(0.8 * rows)
rows_test = rows - rows_train

cmp_embeddings = np.array(df.iloc[:int((0.8 * (len(data['c1'])))), 104:154], dtype='float32')

# trainX - np array of float32, 80% of the data
trainX = np.array(df.iloc[:int((0.8 * (len(data['c1'])))), 4:104], dtype='float32')
trainY = np.array(df.iloc[:int((0.8 * (len(data['c1'])))), 104:154], dtype='float32')

testX = np.array(df.iloc[int((0.8 * (len(data['c1'])))):, 4:104], dtype='float32')
testY = np.array(df.iloc[int((0.8 * (len(data['c1'])))):, 104:154], dtype='float32')


def normalize(array):
    for i in range(len(array)):
        array[i] += 5
        array[i] = array[i] / 10
    return array


def denormalize(array):
    # list = np.array( array, dtype='float32' )
    # print (list)

    for i in range(len(array)):
        array[i] *= 2
        # array[i] -= 5

    return array


'''
trainX = normalize(trainX)
testX = normalize(testX)

trainY = normalize(trainY)
testY = normalize(testY)

print('')
print( 'trainX', trainX.shape )
print( 'trainY', trainY.shape)
print( 'testX', testX.shape )
print( 'testY', testY.shape )
print('')
print( 'trainX', trainX )
'''

# Network building
net = tflearn.input_data([None, 2 * dims])
# print(net.shape)

net = tflearn.fully_connected(net, 128, activation='linear')

net = tflearn.reshape(net, new_shape=[-1, 2, 64])
net = tflearn.lstm(net, 128, dropout=0.8)
# print(net.shape)

net = tflearn.reshape(net, new_shape=[-1, 2, 64])
net = tflearn.lstm(net, 128, dropout=0.8)
# print(net.shape)

net = tflearn.fully_connected(net, 50, activation='linear')

#net = tflearn.batch_normalization(net, beta=0.0, gamma=2.0, trainable=False)

net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001, loss='mean_square')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=0.25, show_metric=True, batch_size=16)

result = model.evaluate(testX, testY)
print("test acc:", result)

# Visible Testing
samples = 3
print("Generate predictions for ", samples, " samples")
predictions = model.predict(testX[:samples])
print("predictions shape:", predictions.shape)
print('')
print('')

'''
print('TRUE EMBEDDING')
for i in range(samples):
    print(df.loc[rows_test+i, ['c1', 'c2', 'cmp']], find_closest_embeddings( testY[i] )[:5])
print('')
# '''

print('PREDICTED EMBEDDING')
for i in range(samples):
    print(df.loc[rows_test + i, ['c1', 'c2', 'cmp']], find_closest_embeddings(predictions[i])[:5])
print('')

# print( find_closest_embeddings( embeddings_dict['hello'])[:5] )

# '''
print('')
for i in range(samples):
    print('True Embedding:')
    print(testY[i])
    print('')
    print('Predicted Embedding:')
    print(predictions[i])
    print('')
# '''

# denormalize(predictions[i])
