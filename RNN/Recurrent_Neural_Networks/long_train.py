#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 06:32:17 2020

@author: gbson
"""

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

# scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

dataset_scaled = sc.fit_transform(dataset)

y1 = dataset_scaled[: ,3]

X1 = dataset_scaled[: ,3]


# separando testes
y_test = y1[len(y1) - 120:]
X_test = X1[len(X1) - 180:]



y = y1[:len(y1) - 200]
X = X1[:len(X1) - 200]


X_train = [] 
y_train = [] 
for i in range(60, len(X)):
    X_train.append(X[i-60:i]) # me pergunto se os primeiros 60 valores de y são usados
    # já que não há 60 valores de X atrás deles
    y_train.append(y[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50)) 
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1)) 
regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')

start = time.time()
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
end = time.time()

demora = end - start
print("a rede demorou {} segundos e {} minuto(s) para ser treinada".format(demora, minute = int(demora / 60)))

inputs = X_test
inputs = inputs.reshape(-1,1)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

real_stock_price = y_test

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.savefig('msft_long_train.png')

# calculating the accuracy of the model
predicted_dia_anterior = []
for i in range(0, len(predicted_stock_price) - 1):
    predicted_dia_anterior.append(predicted_stock_price[i])
predicted_dia_anterior = np.array(predicted_dia_anterior)

predicted_dia_posterior = []
for i in range(1, len(predicted_stock_price)):
    predicted_dia_posterior.append(predicted_stock_price[i])
predicted_dia_posterior = np.array(predicted_dia_posterior)

var_sd = predicted_dia_posterior - predicted_dia_anterior

subiu_desceu_predicted = []
for i in range(0, len(var_sd)):
    if (var_sd[i] > 0):
        subiu_desceu_predicted.append(1)
    elif (var_sd[i] < 0):
        subiu_desceu_predicted.append(0)
    elif (var_sd[i] == 0):
        subiu_desceu_predicted.append('no variance')



predicted_dia_anterior_real = []
for i in range(0, len(real_stock_price) - 1):
    predicted_dia_anterior_real.append(real_stock_price[i])
predicted_dia_anterior_real = np.array(predicted_dia_anterior_real)

predicted_dia_posterior_real = []
for i in range(1, len(real_stock_price)):
    predicted_dia_posterior_real.append(real_stock_price[i])
predicted_dia_posterior_real = np.array(predicted_dia_posterior_real)

var_sd_real = predicted_dia_posterior_real - predicted_dia_anterior_real

subiu_desceu_real = []
for i in range(0, len(var_sd_real)):
    if (var_sd_real[i] > 0):
        subiu_desceu_real.append(1)
    elif (var_sd_real[i] < 0):
        subiu_desceu_real.append(0)
    elif (var_sd_real[i] == 0):
        subiu_desceu_real.append('no variance')
        
acc = []
for i in range(0, len(subiu_desceu_real)):
    if (subiu_desceu_real[i] == subiu_desceu_predicted[i]):
        acc.append(1)
    else:
        acc.append(0)

total_acc = (sum(acc) / len(acc)) * 100
print('a precisão deste modelo é de {}%'.format(total_acc))

a1 = end - start
minute = int(a1 / 60)

print("A RNN demorou {} segundos ou {} minuto(s) para treinar".format(a1, minute))


# Tunning the RNN
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_regressor(optimizer):
    regressor = Sequential()

    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 83))) # 83 inputs
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50)) # não retorna nenhum valor para o início da NN
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1)) # units = 1 | pois só queremos 1 output

    regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')
    return regressor
# loss = 'mean_squared_error' | pois estamos fazendo uma regressão

# Keras nos recomenda RMSprop como optimizer de RNNs, porém 'adam' tem melhor
# performance neste modelo
regressor = KerasClassifier(build_fn = build_regressor)
parameters = {'batch_size': [12, 24, 36, 48, 60],
              'epochs': [100, 250, 500, 750, 1000],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10)
start1 = time.time()
grid_search = grid_search.fit(X_train, y_train)
end1 = time.time()

tunning_time = end1 - start1

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

minute1 = int(tunning_time / 60)

print('o tunning levou {} segundos e {} minuto(s) para completar'.format(tunning_time, minute1))

print('best parameters = {}'.format(best_parameters))
print('best accuracy = {}'.format(best_accuracy))
