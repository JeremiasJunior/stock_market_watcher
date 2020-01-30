#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 02:28:05 2020

@author: gbson
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stockstats import StockDataFrame as Sdf
import pandas_datareader.data as web
import os
import time

os.chdir('/home/gbson/Downloads/Recurrent_Neural_Networks/indicators_research')

msft = pd.read_pickle('MSFT.pkl')

stock = Sdf.retype(msft)

# volume delta against previous day
stock['volume_delta']

# open delta against next 2 day
stock['open_2_d']

# open price change (in percent) between today and the day before yesterday
# 'r' stands for rate.
stock['open_-2_r']

# CR indicator, including 5, 10, 20 days moving average
stock['cr']
stock['cr-ma1']
stock['cr-ma2']
stock['cr-ma3']

# volume max of three days ago, yesterday and two days later
stock['volume_-3,2,-1_max']

# volume min between 3 days ago and tomorrow
stock['volume_-3~1_min']

# KDJ, default to 9 days
stock['kdjk']
stock['kdjd']
stock['kdjj']

# three days KDJK cross up 3 days KDJD
'''
stock['kdj_3_xu_kdjd_3']
'''

# 2 days simple moving average on open price
stock['open_2_sma']

# MACD
stock['macd']
# MACD signal line
stock['macds']
# MACD histogram
stock['macdh']

# bolling, including upper band and lower band
stock['boll']
stock['boll_ub']
stock['boll_lb']

# close price less than 10.0 in 5 days count
stock['close_10.0_le_5_c']

# CR MA2 cross up CR MA1 in 20 days count
stock['cr-ma2_xu_cr-ma1_20_c']

# count forward(future) where close prise is larger than 10
stock['close_10.0_ge_5_fc']

# 6 days RSI
stock['rsi_6']
# 12 days RSI
stock['rsi_12']

# 10 days WR
stock['wr_10']
# 6 days WR
stock['wr_6']

# CCI, default to 14 days
stock['cci']
# 20 days CCI
stock['cci_20']

# TR (true range)
stock['tr']
# ATR (Average True Range)
stock['atr']

# DMA, difference of 10 and 50 moving average
stock['dma']

# DMI
# +DI, default to 14 days
stock['pdi']
# -DI, default to 14 days
stock['mdi']
# DX, default to 14 days of +DI and -DI
stock['dx']
# ADX, 6 days SMA of DX, same as stock['dx_6_ema']
stock['adx']
# ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']
stock['adxr']

# TRIX, default to 12 days
stock['trix']
# MATRIX is the simple moving average of TRIX
stock['trix_9_sma']
# TEMA, another implementation for triple ema
stock['tema']

# VR, default to 26 days
stock['vr']
# MAVR is the simple moving average of VR
stock['vr_6_sma']



# ticker name
ticker = 'MSFT'




# manual tunning

timestep = 30









# data preprocessing


# filtrando na's

stock.isnull().values.any()
isnull = stock.isnull().any()

stock_clean = stock.iloc[10:len(stock), :]

stock_clean.isnull().values.any()
isnull = stock_clean.isnull().any()

stock_clean = stock_clean.iloc[:len(stock_clean) - 2, :]

stock_clean.isnull().values.any()
isnull = stock_clean.isnull().any()


# removendo colunas indesejadas
stock_clean.drop([ 'close_10.0_le', 'close_10.0_le_5_c', 'cr-ma1_20_c', 'cr-ma2_xu_cr-ma1_20_c', 'close_10.0_ge', 'close_10.0_ge_5_fc'],axis=1, inplace=True) # 1 = colunas | 0 = linhas
dataset = stock_clean
dataset = dataset.values

# creating the benchmark dataset
dataset_benchmarked = dataset[:100, :]

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

dataset_benchmarked_scaled = sc.fit_transform(dataset_benchmarked)


# separando os datasets
y = dataset_benchmarked_scaled[: ,3]
X = dataset_benchmarked_scaled

close = dataset[: ,3]
close = close.reshape(-1,1)
close_scaled = sc.fit_transform(close)

# Creating a data structure with 60 timesteps and 1 output
X_train = [] # coleção de 60 dias antes do y
y_train = [] # close
for i in range(timestep, len(X)):
    X_train.append(X[i-timestep:i, :])
    y_train.append(y[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 83)) # 83 = número de colunas



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 83))) # 83 inputs
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50)) # não retorna nenhum valor para o início da NN
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1)) # units = 1 | pois só queremos 1 output

# Compiling the RNN
regressor.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')
# loss = 'mean_squared_error' | pois estamos fazendo uma regressão

# Keras nos recomenda RMSprop como optimizer de RNNs, porém 'adam' tem melhor
# performance neste modelo

# Fitting the RNN to the Training set
start = time.time()
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
end = time.time()
end - start
# batch_size = número de dados forward-propagated antes de ocorrer uma back-propagation


inputs = dataset_benchmarked[len(dataset_benchmarked) - timestep - 30:]
'''
inputs.reshape(-1 ,1)
'''
inputs = sc.transform(inputs)

X_test = []
for i in range(30, len(inputs)):
    X_test.append(inputs[i-30:i, :])
X_test = np.array(X_test)

'''
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 83))
'''
# o código acima é desnecessário parece

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # este
# comando desescalona os dados escalonados.

y_test = dataset_benchmarked[len(dataset_benchmarked) - timestep:, 3]



# visualizando os resultados
plt.plot(y_test, color = 'red', label = 'Preço real')
plt.plot(predicted_stock_price, color = 'blue', label = 'Preço predito')
plt.title('{} Stock Price Prediction'.format(ticker))
plt.xlabel('Time')
plt.ylabel('{} Stock Price'.format(ticker))
plt.legend()
plt.show

real_stock_price = y_test

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
total_acc

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

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



# Fitting the RNN to the Training set
start = time.time()
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
end = time.time()
end - start

