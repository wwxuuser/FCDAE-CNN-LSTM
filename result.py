# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:18:39 2020

@author: 24687
"""


import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
# from RBM import RBMhidden,RBMvisible
from keras.models import load_model
import keras.backend as K
from keras.layers import Dense, Dropout, LSTM,BatchNormalization, Activation,Conv1D,multiply,Flatten
# from Attention import Attention_layer
import time


output_path = 'train_FD001.h5'

sequence_length = 31
test_data = pd.read_csv("test_FD001.csv")
n_turb = test_data['id'].unique().max()

# pick the feature columns
sequence_cols = ['s2', 's3','s4', 's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

# We pick the last sequence for each id in the test data
seq_array_test_last = [test_data[test_data['id']==id][sequence_cols].values[-sequence_length:]
                       for id in range(1, n_turb + 1) if len(test_data[test_data['id']==id]) >= sequence_length]

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

print("This is the shape of the test set: {} turbines, {} cycles and {} features.".format(
    seq_array_test_last.shape[0], seq_array_test_last.shape[1], seq_array_test_last.shape[2]))

print("There is only {} turbines out of {} as {} turbines didn't have more than {} cycles.".format(
    seq_array_test_last.shape[0], n_turb, n_turb - seq_array_test_last.shape[0], sequence_length))


def gated_activation(x):
    # Used in PixelCNN and WaveNet
    tanh = Activation('tanh')(x)
    sigmoid = Activation('sigmoid')(x)
    return multiply([tanh, sigmoid])


y_mask = [len(test_data[test_data['id']==id]) >= sequence_length for id in test_data['id'].unique()]
label_array_test_last = test_data.groupby('id')['RUL'].nth(-1)[y_mask].values

label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def exps(y_true, y_pred):
    return K.mean(K.exp(K.abs(y_pred - y_true)/10), axis=-1) #

def score_calc(y_true, y_pred):
    return K.mean((-1+K.exp(K.relu(y_true-y_pred)/13)+-1+K.exp(K.relu(y_pred-y_true)/10)),axis=-1)


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

# if best iteration's model was saved then load and use it
if os.path.isfile(output_path):
    estimator = load_model(output_path, custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                        'score_calc':score_calc,'RMSE':RMSE,
                                                        'gated_activation':gated_activation,
                                                        'exps':exps})
    startTime = time.time()
    y_pred_test = estimator.predict(seq_array_test_last)
    endTime = time.time()
    print(endTime - startTime)

    y_true_test = label_array_test_last


    # test metrics
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last,verbose = 2)
    print('\n{}'.format(scores_test[0]))
    print('{}'.format(RMSE(y_true_test, y_pred_test)))
    print('{}'.format(scorecalc(y_true_test, y_pred_test)))

    s1 = ((y_pred_test - y_true_test)**2).sum()
    moy = y_pred_test.mean()
    s2 = ((y_pred_test - moy)**2).sum()
    s = 1 - s1/s2
    print('{}%'.format(s * 100))

    test_set = pd.DataFrame(y_pred_test)
    test_set.to_csv('FD001_submit_test.csv', index = None)


    # plt.figure(1, figsize=(15,9))
    # plt.title('actual and prediction with increasing RUL')
    # plt.plot(label_array_test_last, label='actual')
    # plt.legend(loc='upper left')
    # plt.plot(y_pred_test, label='prediction')
    # plt.legend(loc='upper left')
    # plt.show()


