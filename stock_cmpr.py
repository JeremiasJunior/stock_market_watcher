#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:02:06 2019

@author: jeremiasjunior
"""

import numpy as np
from yahoofinancials import YahooFinancials as yf
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
#sys.tracebacklimit = 0

stock_ticker = [] #abreviação dos atívos
date = ['2013-01-01', '2019-09-01']
interval = ['daily', 'monthly']
_interval = 0

#open stock_stickers
def ticker_read(ticker_list):
    ticker_data_open = open('stock_tickers.csv', 'r')
    for i in ticker_data_open.readlines():
        stock_ticker.append(i.strip('\n'))
    ticker_data_open.close()
ticker_read(stock_ticker)

stock_historical_data = list()

#stock_historical_data = (yf('AMD').get_historical_price_data('2019-01-01', '2019-05-05', 'daily'))

for i in stock_ticker:
    stock_historical_data.append(yf(i).get_historical_price_data(date[0], date[1], interval[_interval]))

print('test1')

stock_prices = np.array([])
stock_prices_sindex = []
size_aux = 0
for i in np.arange(len(stock_ticker)):
    for j in stock_historical_data[i][stock_ticker[i]]['prices']:
        size_aux = size_aux + 1
        stock_prices = np.append([j['close']], stock_prices, axis=0)
    stock_prices_sindex.append(size_aux)
    size_aux = 0
print('test2')

stock_prices = np.flip(stock_prices)

def stock_return(stock_list, stock_lsize, stock_n): #organiza o vetor unidimensional do preço das ações
    sum_lsize = int()

    if(stock_n == 0):
        return stock_list.tolist()[0:stock_lsize[stock_n]]
    else:
        for i in range(stock_n+1):
            sum_lsize = stock_lsize[i] + sum_lsize
        return stock_list.tolist()[sum_lsize - stock_lsize[stock_n]:sum_lsize]
    
#print(stock_return(stock_prices, stock_prices_sindex, 0))

stock_lin_regress = list()

#stock_lin_regress.append(stats.linregress(stock_prices[0],stock_prices[1]))

for i in np.arange(len(stock_ticker)):
    plt.figure(i)
    plt.title(str(stock_ticker[i])+"\n" + str(date[0]) + ' to ' +str(date[1]))
    plt.plot(np.arange(len(stock_return(stock_prices, stock_prices_sindex, i))), stock_return(stock_prices, stock_prices_sindex, i))
    plt.grid(True)
    plt.show()
#da pra pegar a intercessão das datas.
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(stock_return(stock_prices, stock_prices_sindex, 0), 
            stock_return(stock_prices, stock_prices_sindex, 1),
            stock_return(stock_prices, stock_prices_sindex, 2))
plt.show()
