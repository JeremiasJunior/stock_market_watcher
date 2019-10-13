#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import stock_cmpr
import numpy as np

import matplotlib.pyplot as plt
import sys

date = ['2011-01-01', '2019-01-01']



ticker_list = stock_cmpr.load_ticker_list('stock_tickers.csv')

historical_data = stock_cmpr.load_historical_data_from_yf(ticker_list, 'yf_historical_data.txt', date, 'monthly')


for i in np.arange(len(ticker_list)):
    price_high = stock_cmpr.get_ticker_historical_data(historical_data, ticker_list, i, 'high')
    price_low = stock_cmpr.get_ticker_historical_data(historical_data, ticker_list, i, 'low')
    price_open = stock_cmpr.get_ticker_historical_data(historical_data,ticker_list, i, 'open')
    price_close = stock_cmpr.get_ticker_historical_data(historical_data,ticker_list, i, 'close')
    price_volume = stock_cmpr.get_ticker_historical_data(historical_data, ticker_list,i, 'volume')
    price_adj = stock_cmpr.get_ticker_historical_data(historical_data,ticker_list, i, 'adjclose')
    price_date = stock_cmpr.get_ticker_historical_data(historical_data,ticker_list, i, 'formatted_date')


    plt.figure(i)
    plt.subplot(211)
    plt.title(str(ticker_list[i])+'\n'+ str(price_date[0])+ ' to '+str(price_date[len(price_date)-1]))
    plt.plot(np.arange(len(price_high)), price_high, color='green', label= "high")
    plt.plot(np.arange(len(price_low)), price_low, color='red', label="low")
    plt.plot(np.arange(len(price_adj)), price_adj, color='blue', ls='--', label='adj')
    plt.plot(np.arange(len(price_open)), price_open, color='pink', ls='--', label='open')
    plt.plot(np.arange(len(price_close)), price_close, color='black', ls='--', label='close')
    plt.legend(loc='upper right')

    plt.grid(True)

    plt.subplot(212)
    plt.plot(np.arange(len(price_volume)), price_volume, color='red')
    plt.grid(True)
    plt.savefig('figure'+str(i))


plt.show()
