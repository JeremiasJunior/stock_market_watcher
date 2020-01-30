#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 06:32:17 2020

@author: gbson
"""



import time

timesteps = int(input("insira o número de timesteps desejado: "))
indicators = 1
drop = float(input("\ninsira o float correspondente ao drop da rede: "))
optimizer = str(input("\ninsira o nome do optimizer que será utilizado: "))
loss = str(input("\ninsira a 'loss' que será utilizada nesta rede: "))
epochs = int(input("\ninsira o número de 'epochs' que esta rede realizará: "))
batch_size = int(input("\ninsira o número que corresponda à batch_size da rede: "))
size_of_test_set = int(input("\ninsira o número que corresponderá ao tamanho do 'test_set' da rede: "))
units = int(input("\nnúmero de 'units' para a rede neural: "))
n_lstm_layers = int(input("\ninsira o número de neurônios LSTM para a rede: "))

start01 = time.time()
acc_list = []

ticker = str(input("Qual ticker iremos analizar agora? (insira ticker):  "))

name_of_the_output_of_this_file = (f'{ticker}_model_{timesteps}timesteps_{drop}drop_{optimizer}optimizer_{epochs}epochs_{batch_size}batch_size_{size_of_test_set}size_of_test_set_performance')


n_loops = 1

for i in range(0,n_loops):
    
    print("\ninitializing {} out of {} loop(s).\n".format(i,n_loops))
    


    # dataset_size = len(dataset)

    print("'{}' timestep(s); '{}' indicator(s); '{}' dropout; '{}' optimizer; '{}' loss calculation; {} epoch(s) and {} batch_size".format(timesteps,indicators, drop, optimizer, loss, epochs, batch_size))

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    local_do_dataset = str(input("\ninsira o caminho para o local onde se encontra o dataset para ser usado na rede: "))
    
    os.chdir(local_do_dataset)
    
    nome_do_arq_do_dataset = str(input("\ninsira o nome do arquivo do dataset: "))
    # preprocessing
    MSFT = pd.read_csv(nome_do_arq_do_dataset, index_col=False, header=0)

    MSFT.isnull().values.any()

    dataset = MSFT
    
    
    
    
    
    # dataset_size
    dataset_size = len(dataset)
    
    
    
    
    
    
    
    dataset = dataset.iloc[:dataset_size,:]

    # training set
    data = dataset.iloc[: ,3:4].values


    # tamanho do test_set
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
    
    
    # construindo o modelo de RNN
    regressor = Sequential()
    regressor.add(LSTM(units = units, return_sequences = True, input_shape = (X_train.shape[1], indicators )))
    regressor.add(Dropout(drop))
    
    for i in range(0, n_lstm_layers):
        regressor.add(LSTM(units = units, return_sequences = True))
        regressor.add(Dropout(drop))
    
    regressor.add(LSTM(units = units)) 
    regressor.add(Dropout(drop))
    regressor.add(Dense(units = 1)) 
    regressor.compile(optimizer = optimizer, loss = loss)

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



    plt.plot(real_stock_price, color = 'red', label = (f'Real {ticker} Stock Price'))
    plt.plot(predicted_stock_price, color = 'blue', label = (f'Predicted {ticker} Stock Price'))
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend()
    plt.savefig(f'{name_of_the_output_of_this_file}.png')

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
            subiu_desceu_predicted.append(2)



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
            subiu_desceu_real.append(2)
        
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
f = open(name_of_the_output_of_this_file, 'w')
sys.stdout = f

print("\nPrinting report of {}:\n".format(name_of_the_output_of_this_file))
print("\nThis code finished in {} seconds and {} minute(s).\n".format(end_of_train, minute01))

end_of_train1 = end_of_train / len(acc_list)
minute02 = end_of_train1 / 60

print("\nEach train session spend {} seconds and {} minute(s)\n".format(end_of_train1, minute02))

acc_real = sum(acc_list) / len(acc_list)

print("\nRNN com parâmetros:\ntimesteps = {};\nindicators = {};\ndrop = {};\noptimizer = {};\nloss = {};\nepochs = {};\nbatch_size = {};\ndataset_size = {};\nsize_of_test_set = {}\n".format(timesteps, indicators, drop, optimizer, loss, epochs, batch_size, dataset_size, size_of_test_set))
print("\na precisão real da RNN é de {}%.".format(acc_real))

sys.stdout = orig_stdout
f.close()
