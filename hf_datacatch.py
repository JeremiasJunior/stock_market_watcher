import numpy as np
from datetime import datetime
from time import sleep
from yahoofinancials import YahooFinancials as yf
import stock_cmpr
import json
import threading 
#i


ticker_list = stock_cmpr.load_ticker_list('stock_tickers.csv')
delay = []

file_json = [] ## armazena os arquivos

def return_date(): ## retorna lista com data atual
    now = datetime.now()
    date = [now.year, now.month, now.day, now.hour, now.minute]
    return date



for i in range(len(ticker_list)):
    print(i)
    global file_json
    date = datetime.now()
    file_path = './data/'
    file_name = (str(ticker_list[i])+'@'+str(date.year)+':'+ str(date.month) + ':'+ str(date.day)+':'+str(date.hour)+':'+ str(date.minute)+'.json')
    file_json.append(open(str(file_path) + str(file_name), 'w'))
    data = {}
    data[0] = (str(ticker_list[i])+'@'+str(date.year)+':'+ str(date.month) + ':'+ str(date.day)+':'+str(date.hour)+':'+ str(date.minute))
    file_json
    with file_json[i] as outfile:
        json.dump(data, outfile)
    return 0

init_data_catcher()
#print(file_json)

f1 = open('test.txt', 'w')
print(f1.closed)

print(file_json[0].closed)
iterator = 0
for t in range(len(ticker_list)):

    finance = yf(ticker_list[t])
    summary = finance.get_summary_data()
    now = datetime.now()
    summary = finance.get_summary_data()

    data = {}
    data['time'] = return_date()
    #data['curr_price'] = finance.get_current_price()
    #data['curr_volume'] = finance.get_current_volume()
    #data['delta_price'] = finance.get_current_change()
    #data['curr_bid'] = summary[ticker_list[t]]['bid']
    #data['curr_ask'] = summary[ticker_list[t]]['ask']

    #print (file_json[t])

    with file_json[t] as outfile:
        print(outfile)
"""
    now = datetime.now(),
    summary = finance.get_summary_data(),
    curr_price = finance.get_current_price(),
    curr_volume = finance.get_current_volume(),
    delta_price = finance.get_current_change(),
    curr_bid = summary[t]['bid'],
    curr_ask = summary[t]['ask'],
"""
