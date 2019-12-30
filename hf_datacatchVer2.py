'''
notas do desenvolvimento

12/13/2019 - 3:28
-primeiro vou fazer tudo o mais direto possivel, depois me preocupo com multithread nas funções

'''


import time
import json
import concurrent.futures
import collections
import numpy as np
import stock_cmpr
from yahoofinancials import YahooFinancials as yf
from datetime import datetime

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

print(data_dict)
print('\n-----script inicializado----')
#fim da inicialização


def get_data(ticker): #c   ta sem atributos de entrada/ lendo direto da variavel global

    now = datetime.now()
    finance = yf(ticker)        
    summary = finance.get_summary_data()

    data_dict[ticker]['curr_price'].append(finance.get_current_price())
    data_dict[ticker]['curr_volume'].append(finance.get_current_volume())
    data_dict[ticker]['delta_price'].append(finance.get_current_change())
    data_dict[ticker]['curr_bid'].append(summary[ticker]['bid'])
    data_dict[ticker]['curr_ask'].append(summary[ticker]['ask'])
    data_dict[ticker]['curr_date'].append([now.month, now.day, now.minute, now.second])
    return str(ticker) + ' finish...'
perf_s = time.perf_counter()

for i in range(10):
    print(i)
    l_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        ticker = ticker_list
        run = [executor.submit(get_data, ticker) for ticker in ticker_list]
        for f in concurrent.futures. as_completed(run):
            print(f.result())
    l_finish = time.perf_counter()
    print(f'tempo de lote {round(l_finish-l_start,2)} segundos...')

perf_f = time.perf_counter()

print(data_dict)
json.dump(data_dict, f_json, indent=2)
print(f'\n\nscript finalizado tempo {round(perf_f-perf_s, 2)} segundos...')
json.dump(data_dict, f_json, )

exit(0)