#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:04:53 2020

@author: gbson
"""


import sys
orig_stdout = sys.stdout
f = open('MSFT_analysis_5_real_deal', 'w')
sys.stdout = f


timesteps = 60
indicators = 1
#drop = 0.2
#optmizer = 'RMSprop'
loss = 'mean_squared_error'
#epochs = 50
#batch_size = 10
dataset_size = 500
size_of_test_set = 10

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stockstats import StockDataFrame as Sdf
import pandas_datareader.data as web
import os
import time
'''
os.chdir('/home/gbson/Desktop/')
'''
# preprocessing
MSFT = pd.read_csv('MSFT_full1.csv', index_col=False, header=0)

MSFT.isnull().values.any()

dataset = MSFT

dataset = dataset.iloc[:dataset_size,:]

    # training set
data = dataset.iloc[: ,3:4].values


    # tamanho do test_set
size_of_test_set = 20
size_of_training_set = len(data) - size_of_test_set




training_set = data[:size_of_training_set]

# scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)

X_train = [] 
y_train = [] 
for i in range(timesteps, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timesteps:i]) 
    y_train.append(training_set_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], indicators)) 


# Tunning the RNN
print("\n initializing RNN Tunning... \n")
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_regressor(optimizer):
    regressor = Sequential()

    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) # 83 inputs
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50)) # não retorna nenhum valor para o início da NN
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1)) # units = 1 | pois só queremos 1 output

    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
    return regressor
# loss = 'mean_squared_error' | pois estamos fazendo uma regressão

# Keras nos recomenda RMSprop como optimizer de RNNs, porém 'adam' tem melhor
# performance neste modelo
regressor = KerasClassifier(build_fn = build_regressor)
parameters = {'batch_size': [24, 32],
              'epochs': [50, 100, 150, 250],
              'optimizer': ['adam', 'rmsprop']}

print('tunning parameters =', parameters)

grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 1)
start1 = time.time()
grid_search = grid_search.fit(X_train, y_train)
end1 = time.time()

tunning_time = end1 - start1

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

minute1 = int(tunning_time / 60)

print('o tunning levou {} segundos ou {} minuto(s) para completar'.format(tunning_time, minute1))

print('best parameters = {}'.format(best_parameters))
print('best accuracy = {}'.format(best_accuracy))

sys.stdout = orig_stdout
f.close()
