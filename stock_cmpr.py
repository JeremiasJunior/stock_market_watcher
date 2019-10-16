#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    instructions:
        You must execute in the following order:
            1-load_ticker_list
            2-load_historical_data_from_yf
            
            then you are ready to call get_ticker_historical_data
"""
import numpy as np
from yahoofinancials import YahooFinancials as yf

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

def load_historical_data_from_yf(ticker_list, file_out, date, interval):
    """
        description:
            gets historical price of a set of tickers.

        parameters: 
            ticker_list: takes a list with all desired tickers
            file_out: saves the JSON output to a file.
            date: list containing the initial date and final date i.e ['2008-01-01', '2018-01-01']
            interval: must contein the interval which you want data i.e 'monthly' or 'daily'
    """
    stock_historical_data = []
    for i in ticker_list:
        stock_historical_data.append(yf(i).get_historical_price_data(date[0], date[1], interval))

    return stock_historical_data

def get_ticker_num(ticker_list, ticker):
    """
        parameters:
            ticker_list: list containing tickers
            ticker: string of the specific ticker
    """
    count = 0 
    for i in ticker_list:
        if(i == ticker):
            return count
        count = count + 1
    return -1

def get_ticker_historical_data(historical_data, ticker_list ,ticker_num, price_parameter):
    """
        parameters:
            historical_data: dict
            ticker_list : list containing all tickers
            ticker: sting
            price_parameter: can be 'high', 'low', 'open', 'close', 'volume', 'adjclose'
    """
    stock_prices = np.array([])


    for j in historical_data[ticker_num][ticker_list[ticker_num]]['prices']:
        stock_prices = np.append([j[price_parameter]], stock_prices, axis = 0)
    stock_prices = np.flip(stock_prices)

    return stock_prices

def delta_list(l_list):
    aux = 0
    size_list = len(l_list)
    new_list = list()
    new_list.append(0)

    for i in l_list:
        if(aux == (size_list-2)):
            new_list.append(0)
            return new_list
        new_list.append((i - new_list[aux]))
        aux = aux+1
    return new_list
    

def avg_between_lists(a_list, b_list):
    new_list = []
    aux = 0
    for i in a_list:
        new_list.append((a_list[aux] + b_list[aux])/2)
        aux = aux+1
    return new_list

def log_normalization(l_list):
    new_list = []

    for i in l_list:
        new_list.append(np.log(i))

    return new_list
