#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
notas do desenvolvimento

12/30/2019 - 3:28
-primeiro vou fazer tudo o mais direto possivel, depois me preocupo com multithread nas funções

-12/31/2019 - 14:18
resultado dos testes com 100 tickers:
-altamente dependente de quantidade de threads, no meu computador terminei 6 lotes de 100 acoes em 278.28 segundos, o servidor de 1 processador da hetzner levou 810.36 segundos
-a quantidade de dados gerada num dia não deve passar de uns 2 mb

segundo resultado com o VPS de 2 cores, reduziu o tempo pra 413.82 segundos e meu pc ficou em 206.13 segundos

teste com 50 tickers:
na vps de 2 cores
no meu pc 122.91 segundos
na vps z 202.81 segundos

teste no azure com 100 tickers
meu pc 179.17 segundos
cps 4 209.45

'''


import time
import json
import concurrent.futures
import collections
import numpy as np
import stock_cmpr
from yahoofinancials import YahooFinancials as yf
from datetime import datetime
import yahoofinancials as YF
'''
inicializa o basico:

    -leitura dos tickers
    -cria arquivo .json
    -inicializa a variavel dict

'''

ticker_list = stock_cmpr.load_ticker_list('stock_tickers.csv')
now = datetime.now()
f_name = (str(now.year) + '-' + str(now.month)+'-'+str(now.day)+'@'+str(now.hour)+'-'+str(now.minute))
f_dir = './data/'+str(f_name)+'.json'
f_json = open(f_dir,'w+')

data_dict = collections.defaultdict(list)

for ticker in ticker_list:
    data_dict[ticker] = {
        'curr_price': [],
        'curr_volume': [],
        'delta_price': [],
        'curr_bid': [],
        'curr_ask': [],
        'curr_date': []
    }

print('\n-----script inicializado----')
#fim da inicialização

perf_s = time.perf_counter()
tl_size = len(ticker_list)
t_time = float()

#new_file = open('ticker8.csv', 'w')

def data_check(ticker): #c   ta sem atributos de entrada/ lendo direto da variavel global
    now = datetime.now()
    finance = yf(ticker)
    #summary = finance.get_summary_data()

    if(finance.get_current_price() < 8):
        new_file.write(str(ticker) + '\n')

    return ticker



def get_data(ticker): #c   ta sem atributos de entrada/ lendo direto da variavel global
    now = datetime.now()
    try:
        finance = yf(ticker)
    except:
        print(f"\n---err 404: {ticker} finance---\n")
        return 0
    try:
        summary = finance.get_summary_data()
    except:
        print(f"\n---err 404: {ticker} summary---\n")
        return 0
    try:
        data_dict[ticker]['curr_price'].append(finance.get_current_price())
    except:
        print(f"\n---err 404: {ticker} curr_price---\n")
        return 0
    try:
        data_dict[ticker]['curr_volume'].append(finance.get_current_volume())
    except:
        print(f"\n---err 404: {ticker} curr_volume---\n")
        return 0
    try:
        data_dict[ticker]['delta_price'].append(finance.get_current_change())
    except:
        print(f"\n---err 404: {ticker} delta_price---\n")
        return 0
    try:
        data_dict[ticker]['curr_bid'].append(summary[ticker]['bid'])
    except:
        print(f"\n---err 404: {ticker} curr_bid---\n",ticker)
        return 0
    try:
        data_dict[ticker]['curr_ask'].append(summary[ticker]['ask'])
    except:
        print(f"\n---err 404: {ticker} curr_ask---\n")
        return 0
    try:
        data_dict[ticker]['curr_date'].append([now.minute, now.second])
    except:
        print(f"\n---err 404: {ticker} curr_date---\n")
        return 0

    return ticker

perf_s = time.perf_counter()
tl_size = len(ticker_list)
t_time = float()

print(now.hour)
iterator = 0
#for _ in range(1):
while(now.hour > 9 and now.hour < 21):

    print('\ninicializando lote...\n' + str(iterator) + '\n')
    l_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        ticker = ticker_list
        try:
            run = [executor.submit(get_data, ticker) for ticker in ticker_list]
        except:
            print('--err master 404--')
            pass
        for f in concurrent.futures.as_completed(run):
            print(str(f.result()) + ' of ticker ' + str(tl_size))
            tl_size -= 1

    l_finish = time.perf_counter()
    print(f'\nfinalizando lote...\ntempo de lote {round(l_finish-l_start,2)} segundos...')
    tl_size = len(ticker_list)
    iterator += 1

perf_f = time.perf_counter()
print(f'\n\nscript finalizado tempo {round(perf_f-perf_s, 2)} segundos...')

json.dump(data_dict, f_json, indent=2)

exit(0)