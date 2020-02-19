#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 20:30:25 2020

@author: gbson
"""

# Recurrent Neural Networks

# Importing Libraries:
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# changing working directory
os.chdir('/home/gbson/Downloads/Recurrent_Neural_Networks/')

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values # numpy array da
# variável 'open' do dataset.

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler # para normalização
sc = MinMaxScaler(feature_range = (0, 1)) # todos os nossos
# preços serão escalonados entre 0 e 1 (utilizando o método da
# normalização).
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# X_train = [] -> uma lista vazia
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    # no código acima, pegaremos todos os dados escalonados históricos 60 dias
    # antes de x dia para analisarmos alguns padrões e colocaremos num vetor.

X_train, y_train = np.array(X_train), np.array(y_train)
# o código acima organiza os vetores do loop 'for' acima em umas colunas aí

# X_train = 'valor de open de 60 dias antes do valor do qual queremos prever
# y_train = 'valor que queremos prever com X_train'



# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_train.shape[0] = número de linhas
# X_train.shape[1] = número de 'timesteps' (colunas)
# 1 = input da nossa rede (neste caso, número de indicadores de stock market)

# 3D tensor with shape = (batch_size, timesteps, input_size)
# input_size pode ser os indicadores para o stock_price



# Part 2 -Building the RNN
# Importing the Keras Libraries and Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) # primeira LSTM layer | .add adiciona uma layer
regressor.add(Dropout(0.2)) # 20% dos neurônios da LSTM layer serão ignorados de
# cada 'iteration' do treino

# Adding a second LSTM layer and some Dropout regularization
