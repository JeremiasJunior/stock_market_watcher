# -*- coding: utf-8 -*-

import numpy as np
from yahoofinancials import YahooFinancials as yf
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.stattools import adfuller
sys.tracebacklimit = 0

#input de dados via yahoo financials
bradesco = []
itau = []

itau            = yf('CL=F').get_historical_price_data('2019-05-01', '2019-07-31', 'daily')
bradesco        = yf('INTC').get_historical_price_data('2019-05-01', '2019-07-31', 'daily')


print (bradesco)

itau_plt =[[],[]]
bradesco_plt = [[],[]]

for i in itau['AMD']['prices']:
    itau_plt[0].append(i['formatted_date'])
    itau_plt[1].append(i['close'])
    
for i in bradesco['INTC']['prices']:
    bradesco_plt[0].append(i['formatted_date'])
    bradesco_plt[1].append(i['close'])

 
_xlabel = []
j = 0
for i in bradesco_plt[0]:
    _xlabel = np.linspace

#calcula média aritimética de uma lista
def avg(x_list):
    _avg = 0
    for i in np.arange(len(x_list)):
        _avg += x_list[i]
    return _avg/len(x_list)

#calcula a variancia de uma lista
def variance(x_list):
    _sigma_square = 0
    _avg = avg(x_list)
    for i in np.arange(len(x_list)):
        _sigma_square += pow((x_list[i] - _avg),2)
    return _sigma_square

#calcula o desvio padrão de uma lista 
def std_deviation(x_list):
    return np.sqrt(variance(x_list)/(len(x_list) - 1))
    
#calcula os residuals da regreção
def r_residuals(x_list, y_list):
    #médias
    x_avg = avg(x_list)
    y_avg = avg(y_list)
    
    #desvio padrao
    _S_x = std_deviation(x_list)
    _S_y = std_deviation(y_list)

    #calculo do resíduo
    xy_sum = 0
    for i in np.arange(len(x_list)):
        xy_sum += ((x_list[i]- x_avg)/_S_x)*((y_list[i] - y_avg)/_S_y)
    return xy_sum * (1/(len(x_list)-1))

#valor do coeficiente de co-relação
r_residual = r_residuals(itau_plt[1], bradesco_plt[1])
print('residual  = ' + str(r_residual))

#desvios padrões das duas ações
S_x = std_deviation(itau_plt[1])
S_y = std_deviation(bradesco_plt[1])

#função da regressão linear 
r_line = []
def r_line_func(x):
    return r_residual*(S_y/S_x)*i + (avg(bradesco_plt[1]) - (r_residual*(S_y/S_x))*avg(itau_plt[1]))

#cria uma lista com os valores da regressão para o plot
for i in np.arange(np.max(itau_plt[1])):
    r_line.append(r_line_func(i))


#lista para plot do grafico dos residuos
residual_list = []
_j = 0 
for i in itau_plt[1]:
    residual_list.append(bradesco_plt[1][_j] - r_line_func(itau_plt[1]))
    _j+=1
    if(_j > len(bradesco_plt[1])):
        _j = 0
        

#plots
        
#valor das ações no tempo
plt.figure(0)
plt.subplot(211)
itau_plot = plt.plot(np.arange(len(itau_plt[1])),itau_plt[1], label='ITUB')
plt.subplot(212)
bradesco_plot = plt.plot(np.arange(len(bradesco_plt[1])),bradesco_plt[1], label='BBD')
plt.legend(loc='best')
#plt.xticks([i*12 for i in range(17)], ['%i'%w for w in range(3,19)])
plt.xlabel('year')
plt.ylabel('price (USD)')
plt.grid(True)
plt.savefig('price.png')
plt.show()

#scatter do preço do ITUBxBBD
plt.figure(1)
plt.scatter(itau_plt[1],bradesco_plt[1], color='red')
plt.plot(np.arange(len(r_line)), r_line, color = 'blue')
plt.title('price (USD)')
plt.xlabel('ITUB')
plt.ylabel('BBD')
#plt.xlim([0,15])
#plt.ylim([0,15])
plt.savefig('BBDxITUB.png')
plt.grid(True)

#calculo do Augmented Dickey-Fuller
dftest = adfuller(residual_list, autolag='AIC')
print ('dftest = ' + str(dftest[1]))
print ('cointegração : ' + str((1-dftest[1]) * 100) + '%' )

#plot dos residuos
plt.figure(2)
plt.plot(np.arange(len(residual_list)), residual_list, color = 'blue')
plt.plot(np.arange(len(residual_list)), np.zeros(len(residual_list)), color='red')
plt.grid(True)
plt.title('residual')
plt.savefig('residual.png')
plt.show()

