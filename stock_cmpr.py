#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from yahoofinancials import YahooFinancials as yf
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
#sys.tracebacklimit = 0

def stock_price_return(stock_list, stock_lsize, stock_n): 
    """
        description:
            to facilitate the access of the stock price array, by indexing in a unidimentional array.
        
        parameters:
            stock_list: the list which conteins the prices.
            stock_lsize: the ammount of prices.
            stock_n: the number of the stock, whitch is in order of the file from where was imported.
        
    """
    sum_lsize = int()

    if(stock_n == 0):
        return stock_list.tolist()[0:stock_lsize[stock_n]]
    else:
        for i in range(stock_n+1):
            sum_lsize = stock_lsize[i] + sum_lsize
        return stock_list.tolist()[sum_lsize - stock_lsize[stock_n]:sum_lsize]

def load_ticker_list(file_in):
    stock_ticker_list = []
    stock_ticker_input_file = open(file_in, 'r')

    for i in stock_ticker_input_file.readlines():
        stock_ticker_list.append(i.strip('\n'))

    stock_ticker_input_file.close

    return stock_ticker_list

def load_historical_data_from_yf(ticker_list, file_out, date, interval, buffer_mode=True):
    """
        description:
            gets historical price of a set of tickers.

        parameters: 
            ticker_list: takes a list with all desired tickers
            file_out: saves the JSON output to a file.
            date: list containing the initial date and final date i.e ['2008-01-01', '2018-01-01']
            interval: must contein the interval which you want data i.e 'monthly' or 'daily'
            buffer_mode: the data will be stored in a file 
    """
    if(buffer_mode == True):
        stock_data_output = open(file_out, 'w')
    stock_historical_data = []
    for i in ticker_list:
        print(i)
        stock_historical_data.append(yf(i).get_historical_price_data(date[0], date[1], interval))
        stock_data_output.close
    
    if(buffer_mode == True):
        for i in np.arange(len(ticker_list)):
            print("saving ", ticker_list[i])
            stock_data_output.writelines(str(stock_historical_data[i]))
            stock_data_output.writelines('\n')
    stock_data_output.close
    return stock_historical_data

    
def get_prices(historical_data, ticker_list, price_parameter):
    """
        description: 
            Returns (and can save into a file ) the prices from the historical data
        parameters:
            historical_data: The Dict imported by load_historical_data_from_yf, or imported from a file.
            ticker_list: a list with all tickers
            price_parameter: can be 'high', 'low', 'open', 'close', 'volume', 'adjclose'
    """
    stock_prices = np.array([])
    stock_prices_sindex = np.array([])
    size_aux = 0
    
    for i in np.arange(len(ticker_list)):
        for j in stock_historical_data[i][ticker_list[i]][price_parameter]:
            size_aux = 0
            stock_prices = np.append([j[price_parameter]], stock_prices, axis=0)
        stock_prices_sindex.append(size_aux)
        size_aux = 0

    stock_prices = np.flip(stock_prices)

    return stock_prices

#print(stock_return(stock_prices, stock_prices_sindex, 0))
#for i in np.arange(len(stock_ticker)):
#    print(len(stock_return(stock_prices, stock_prices_sindex, 0)))


#stock_lin_regress = list()
#stock_lin_regress.append(stats.linregress(stock_prices[0],stock_prices[1]))

#for i in np.arange(len(stock_ticker)):
#    plt.figure(i)
#    plt.title(str(stock_ticker[i])+"\n" + str(date[0]) + ' to ' +str(date[1]))
#    plt.plot(np.arange(len(stock_return(stock_prices, stock_prices_sindex, i))), stock_return(stock_prices, stock_prices_sindex, i))
#    plt.grid(True)
#    plt.show()
