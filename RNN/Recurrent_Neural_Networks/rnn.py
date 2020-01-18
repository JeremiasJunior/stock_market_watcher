# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir('/home/gbson/Downloads/Recurrent_Neural_Networks/')
# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
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
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# loss = 'mean_squared_error' | pois estamos fazendo uma regressão

# Keras nos recomenda RMSprop como optimizer de RNNs, porém 'adam' tem melhor
# performance neste modelo



# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# batch_size = número de dados forward-propagated antes de ocorrer uma back-propagation



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # escalonará os 'inputs' para a escala que os
# dados treinados foram escalonados
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # este
# comando desescalona os dados escalonados.


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

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
