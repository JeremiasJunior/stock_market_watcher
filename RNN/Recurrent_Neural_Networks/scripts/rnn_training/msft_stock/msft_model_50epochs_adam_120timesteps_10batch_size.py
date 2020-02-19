#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 06:32:17 2020

@author: gbson
"""
'''
import sys

# salvando outputs deste script
orig_stdout = sys.stdout
f = open('long_train_little_output.txt', 'w')
sys.stdout = f
'''
name_of_the_output_of_this_file = 'msft_model_50epochs_adam_120timesteps_10batch_size_performance'





import time

timesteps = 120
indicators = 1
drop = 0.2
optmizer = 'adam'
loss = 'mean_squared_error'
epochs = 50
batch_size = 10
size_of_test_set = 20

start01 = time.time()
acc_list = []

n_loops = 1

for i in range(0,n_loops):
    
    print("initializing {} out of {} loop(s).".format(i + 1,n_loops))
    
    timesteps = 120
    indicators = 1 # ou utilize 'training_set.shape[1]' para saber quantos indicadores
    drop = 0.2
    optmizer = 'adam'
    loss = 'mean_squared_error'  
    epochs = 50
    batch_size = 10

    # dataset_size = len(dataset)

    print("'{}' timestep(s); '{}' indicator(s); '{}' dropout; '{}' optmizer; '{}' loss calculation; {} epoch(s) and {} batch_size".format(timesteps,indicators, drop, optmizer, loss, epochs, batch_size))

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from stockstats import StockDataFrame as Sdf
    import pandas_datareader.data as web
    import os
    import time
    
    os.chdir('/home/gbson2/stock_cmpr/RNN/Recurrent_Neural_Networks/datasets/')
    
    # preprocessing
    MSFT = pd.read_csv('MSFT_full1.csv', index_col=False, header=0)

    MSFT.isnull().values.any()

    dataset = MSFT
    
    
    
    
    
    # dataset_size
    dataset_size = len(dataset)
    
    
    
    
    
    
    
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


    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], indicators )))
    regressor.add(Dropout(drop))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(drop))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(drop))
    regressor.add(LSTM(units = 50)) 
    regressor.add(Dropout(drop))
    regressor.add(Dense(units = 1)) 
    regressor.compile(optimizer = optmizer, loss = loss)

    start = time.time()
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
    end = time.time()

    demora = end - start
    minute = int(demora / 60)
    print("a rede demorou {} segundos ou {} minuto(s) para ser treinada".format(demora, minute))


    test_set = data[len(data) - size_of_test_set:]
    real_stock_price = test_set

    # X_test
    inputs = data[len(data) - timesteps - size_of_test_set:]
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(timesteps, len(inputs)):
        X_test.append(inputs[i-timesteps:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], indicators))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)



    plt.plot(real_stock_price, color = 'red', label = 'Real MSFT Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted MSFT Stock Price')
    plt.title('MSFT Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('MSFT Stock Price')
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
    acc_list.append(total_acc)

    print('a precisão deste modelo é de {}%'.format(total_acc))

    a1 = end - start
    minute = int(a1 / 60)

    print("A RNN demorou {} segundos ou {} minuto(s) para treinar".format(a1, minute))

end01 = time.time()

end_of_train = end01 - start01
minute01 = int(end_of_train / 60)

import sys
orig_stdout = sys.stdout
f = open('msft_model_50epochs_adam_120timesteps_10batch_size_performance', 'w')
sys.stdout = f

print("\nPrinting report of {}:\n".format(name_of_the_output_of_this_file))
print("\nThis code finished in {} seconds and {} minute(s).\n".format(end_of_train, minute01))

end_of_train1 = end_of_train / len(acc_list)
minute02 = end_of_train1 / 60

print("\nEach train session spend {} seconds and {} minute(s)\n".format(end_of_train1, minute02))

acc_real = sum(acc_list) / len(acc_list)

print("\nRNN com parâmetros:\ntimesteps = {};\nindicators = {};\ndrop = {};\noptimizer = {};\nloss = {};\nepochs = {};\nbatch_size = {};\ndataset_size = {};\nsize_of_test_set = {}\n".format(timesteps, indicators, drop, optmizer, loss, epochs, batch_size, dataset_size, size_of_test_set))
print("\na precisão real da RNN é de {}%.".format(acc_real))

sys.stdout = orig_stdout
f.close()
