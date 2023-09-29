# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:12:05 2020

@author: tianc
"""

import tensorflow as tf

import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Lambda, Flatten, Conv1D,  BatchNormalization
from keras.layers import LeakyReLU
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import objectives
from keras import regularizers
from keras.models import load_model

import os
import math


# input_path = 'pre_FD001.h5'
output_path = 'train_FD001.h5'
train_data = pd.read_csv('train_FD001.csv')
test_data = pd.read_csv("test_FD001.csv")
n_turb = train_data['id'].unique().max()

# pick a large window size of 30 cycles
sequence_length = 31


# function to reshape features into (samples, time steps, features)
def reshapeFeatures(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length + 1), range(seq_length, num_elements + 1)):
        yield data_matrix[start:stop, :]



sequence_cols = ['s2', 's3', 's4', 's7', 's8',
                 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
# 2, 3, 4, 7, 8, 9,11, 12, 13, 14, 15, 17, 20 and 21


# generator for the sequences
feat_gen = (list(reshapeFeatures(train_data[train_data['id'] == id], sequence_length, sequence_cols)) for id in
            range(1, n_turb + 1))

# generate sequences and convert to numpy array
feat_array = np.concatenate(list(feat_gen)).astype(np.float32)
print("The data set has now shape: {} entries, {} cycles and {} features.".format(feat_array.shape[0],
                                                                                  feat_array.shape[1],
                                                                                  feat_array.shape[2]))


# function to generate label
def reshapeLabel(id_df, seq_length=sequence_length, label=['RUL']):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length - 1: num_elements, :]


# generate labels
label_gen = [reshapeLabel(train_data[train_data['id'] == id]) for id in range(1, n_turb + 1)]
label_array = np.concatenate(label_gen).astype(np.float32)
print(label_array.shape)

seq_array_test_last = [test_data[test_data['id'] == id][sequence_cols].values[-sequence_length:]
                       for id in range(1, n_turb + 1) if len(test_data[test_data['id'] == id]) >= sequence_length]
seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

y_mask = [len(test_data[test_data['id'] == id]) >= sequence_length for id in test_data['id'].unique()]
label_array_test_last = test_data.groupby('id')['RUL'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)


# MODEL

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def mean_absolute_error(y_true, y_pred):
    return K.abs(K.mean(y_pred - y_true, axis=-1))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def exps(y_true, y_pred):
    return K.mean((-1 + K.exp(K.relu(y_true - y_pred) / 13) + -1 + K.exp(K.relu(y_pred - y_true) / 10)),
                  axis=-1) + 0.5 * K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))  #


def score_calc(y_true, y_pred):
    return K.mean((-1 + K.exp(K.relu(y_true - y_pred) / 13) + -1 + K.exp(K.relu(y_pred - y_true) / 10)), axis=-1)


def RMSE(y_true, y_pred):
    RMSE=0
    c=y_pred-y_true
    d=c.shape[0]
    for i in range(0,d,1):
            RMSE=RMSE+c[i]*c[i]
    return math.sqrt(RMSE/d)


def scorecalc(y_true, y_pred):
    score=0
    c=y_pred-y_true
    d=c.shape[0]
    for i in range(0,d,1):
        if c[i]<0:
            score=score-1+math.exp(-c[i]/10)
        else:
            score=score-1+math.exp(c[i]/13)
    return score


nb_features = feat_array.shape[2]
nb_out = label_array.shape[1]

# encoder = load_model(input_path)

x = keras.Input(shape=(sequence_length, nb_features))
print('input: ', x.shape)

x01 = LSTM(units=256, return_sequences=True)(x)
x02 = Dropout(0.2)(x01)
print('LSTM1: ', x02.shape)

x11 = keras.layers.Conv1D(filters=256, kernel_size=2, padding='same', dilation_rate=1)(x)
x12 = LeakyReLU(0.3)(x11)
print('CNN1', x12.shape)
c1 = keras.layers.Add()([x02, x12])
print('add1', c1.shape)

x03 = LSTM(units=64, return_sequences=True)(x02)
x04 = Dropout(0.2)(x03)
print('LSTM2: ', x04.shape)
x13 = keras.layers.Conv1D(filters=64, kernel_size=2, padding='same', dilation_rate=1)(c1)
x14 = LeakyReLU(0.3)(x13)
print('CNN2', x14.shape)

c2 = keras.layers.Add()([x04, x14])
print('add2', c2.shape)

x05 = LSTM(units=32, return_sequences=True)(x04)
x06 = Dropout(0.2)(x05)
print('LSTM3: ', x06.shape)
x15 = keras.layers.Conv1D(filters=32, kernel_size=2, padding='same', dilation_rate=1)(c2)
x16 = LeakyReLU(0.3)(x15)
print('CNN3', x12.shape)

c3 = keras.layers.Add()([x06, x16])
print('add3', c3.shape)


x06 = Flatten()(x06)
print('flatten-LSTM', x06.shape[1])
x07 = Dense(units=16)(x06)
x08 = LeakyReLU(0.3)(x07)
print('dense-LSTM', x06.shape)
c3 = Flatten()(c3)
print('flatten-CNN', c3.shape)
x17 = Dense(units=16)(c3)
x18 = LeakyReLU(0.3)(x17)
print('dense-CNN', x06.shape)

c4 = keras.layers.Add()([x06, c3])
print('add4', c4.shape)

y = keras.layers.Concatenate(axis=1)([x06, c4])
print(y.shape)

fix = keras.Model(x, y, name="fix_model")

# Bidirectional
model = Sequential()
# model.add(encoder)
model.add(fix)
model.add(Dense(units=16))
model.add(Dropout(0.2, name="dropout_4"))
model.add(Dense(units=nb_out))
model.add(LeakyReLU(0.3, name="activation_0"))

model.compile(loss=score_calc, optimizer=Adam(lr=1e-4), metrics=[root_mean_squared_error, mean_squared_error, mean_absolute_error, exps, score_calc])
# 'binary_crossentropy'
print(model.summary())


# fit the network
history = model.fit(feat_array, label_array, epochs=200, batch_size=256, shuffle=True, validation_split=0.1,
                    verbose=1,
                    callbacks=[
                        keras.callbacks.EarlyStopping(monitor='val_score_calc', min_delta=0, patience=20, verbose=0,
                                                      mode='min'),
                        keras.callbacks.ModelCheckpoint(output_path, monitor='val_score_calc', save_best_only=True,
                                                        mode='min', verbose=0)]
                    )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path))


# summarize history for Loss
fig_loss = plt.figure(figsize=(15,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

estimator = load_model(output_path, custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                        'score_calc':score_calc,'RMSE':RMSE,
                                                        'exps':exps})

y_pred_test = estimator.predict(seq_array_test_last)
y_true_test = label_array_test_last


s1 = ((y_pred_test - y_true_test)**2).sum()
moy = y_pred_test.mean()
s2 = ((y_pred_test - moy)**2).sum()
s = 1 - s1/s2
print('{}%'.format(s * 100))


