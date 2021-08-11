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
from scipy import spatial

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

#from tflearn.objectives import cosine_similarity


# from tflearn.data_utils import to_categorical, pad_sequences

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

testY_copy = testY


def normalize(array):
    for i in range(len(array)):
        array[i] += 5
        array[i] = array[i] / 10
    return array


def denormalize(array):
    for i in range(len(array)):
        array[i] *= 10
        array[i] -= 5
    return array

'''
trainX = normalize(trainX)
testX = normalize(testX)

trainY = normalize(trainY)
testY = normalize(testY)

print('Denormalized:')
for i in range(3):
    print( df.loc[ int((0.8 * (len(data['c1']))))+i, ['cmp'] ] )
    print('Original Embedding:')
    print( testY_copy[i] )
    print('')
    print('Denormalized Embedding:')
    print( denormalize(testY[i]) )
    print('')
# '''

'''
print('')
print( 'trainX', trainX.shape )
print( 'trainY', trainY.shape)
print( 'testX', testX.shape )
print( 'testY', testY.shape )
print('')
print( 'trainX', trainX )
'''

# Network building
net = input_data([None, 2 * dims])

#net = embedding(net, input_dim=20000, output_dim=128)

net = tflearn.reshape(net, [-1,2,50])
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
# net = dropout(net, 0.8)

net = tflearn.reshape(net, [-1,2,128])
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
# net = dropout(net, 0.8)

net = tflearn.reshape(net, [-1,2,128])
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
# net = dropout(net, 0.8)

net = tflearn.reshape(net, [-1,2,128])
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
# net = dropout(net, 0.5)

net = tflearn.reshape(net, [-1,2,128])
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128))
# net = dropout(net, 0.8)

net = fully_connected(net, 50, activation='linear')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='mean_square')
# change to cosine similarity

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=60, validation_set=0.25, show_metric=True, batch_size=16)

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
    print(df.loc[rows_train + i, ['c1', 'c2', 'cmp']], find_closest_embeddings( testY[i] )[:5])
print('')
# '''

print('PREDICTED EMBEDDING')
for i in range(samples):
    print(df.loc[rows_train + i, ['c1', 'c2', 'cmp']], find_closest_embeddings( predictions[i] )[:5])
print('')

'''
print('Denormalized:')
for i in range(samples):
    print('True Embedding:')
    print( testY[i] )
    print('')
    print('Predicted Embedding:')
    print( predictions[i] )
    print('')
# '''
