# -*- coding: utf-8 -*-

"""
Check tflearn.org for more information about neural nets
"""
from __future__ import division, print_function, absolute_import

import pandas as pd
import numpy as np
import tflearn

from numpy import random

# np.set_printoptions(precision=5)

random.seed(1)
np.random.seed(1)

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

# Data loading and preprocessing
# from tflearn.data_utils import load_csv
data = pd.read_csv('NoteBooks/data/all_embeddings_forML.csv')

df = data.sample( frac=1, axis=0 ) # Randomize order of the rows in data

# X -> Array with c1_embedding and c2_embedding
# Y -> Array of cmp_embeddings
X = np.array( df.iloc[:int((0.9*(len(data['c1'])))), 4:104], dtype='float32' )
Y = np.array( df.iloc[:int((0.9*(len(data['c1'])))), 104:154], dtype='float32' )

# testX -> last 10% of X
# testY -> last 10% of Y
testX = np.array( df.iloc[int((0.9*(len(data['c1'])))):, 4:104], dtype='float32' )
testY = np.array( df.iloc[int((0.9*(len(data['c1'])))):, 104:154], dtype='float32' )

# Building deep neural network
input_layer = tflearn.input_data( shape=[None,100] )
dense1 = tflearn.fully_connected( input_layer, 100, activation='ReLU', regularizer='L2', weight_decay=0.001)

dropout1 = tflearn.dropout(dense1, 0.8)

dense2 = tflearn.fully_connected(dropout1, 100, activation='ReLU', regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)

reLU = tflearn.fully_connected(dropout2, 50, activation='ReLU')

# softmax = tflearn.fully_connected(reLU, 50, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
# sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
adam = tflearn.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')

# r2_op(y_pred, y_true)
std_error = tflearn.metrics.R2() # USE R2 instead of top k
net = tflearn.regression(reLU, optimizer=adam, metric=std_error, loss='mean_square')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
# print(X.shape,Y.shape,testX.shape,testY.shape)
model.fit(X, Y, n_epoch=20, validation_set=(testX, testY), show_metric=True, batch_size=8, run_id="dense_model")

# Predict a cmp embedding
# print( testX[0] )

x0 = testX[0]
x1 = testX[1]

predY = model.predict([x0, x1])
print("Predicted 1:")
print(predY[0])
print("Predicted 2:")
print(predY[1])

print('')
print("Actual:")
print(testY[0])

print('')
# print(find_closest_embeddings(embeddings_dict['hello'])[:5])
print(find_closest_embeddings( predY[0] )[:5])
print(find_closest_embeddings( predY[1] )[:5])

# print(find_closest_embeddings( testY[0] )[:5])
# print(find_closest_embeddings( testY[1] )[:5])

# print( predY[0].dtype )
