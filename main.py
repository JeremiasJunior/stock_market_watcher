#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import stock_cmpr
import numpy as np
import sys
import plot_module

"""
    ideia para executar:
        fazer o plot de um delta do preço da ação.
        normalizar logaritimicamente o volume de transações
        fazer um scatter tanto do volume normalizado x preço
                                  volume normalizado x delta preço.

"""

date = ['2015-01-01', '2019-01-01']
ticker_list = stock_cmpr.load_ticker_list('stock_tickers.csv')
interval = 'daily'
historical_data = stock_cmpr.load_historical_data_from_yf(ticker_list, 'yf_historical_data.txt', date, interval)

plot_module.plot_price_data(historical_data, ticker_list)
plot_module.scatter_price_data(historical_data, ticker_list)