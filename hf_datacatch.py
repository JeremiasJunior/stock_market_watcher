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


def init_data_catcher():
    for i in range(len(ticker_list)):
        print(i)
        date = datetime.now()
        file_path = './data/'
        file_name = (str(ticker_list[i])+'@'+str(date.year)+':'+ str(date.month) + ':'+ str(date.day)+':'+str(date.hour)+':'+ str(date.minute)+'.json')
        print(file_name)
        file_json.append(open(str(file_path) + str(file_name), 'w'))
        data = {}
        data[0] = (str(ticker_list[i])+'@'+str(date.year)+':'+ str(date.month) + ':'+ str(date.day)+':'+str(date.hour)+':'+ str(date.minute))
        with file_json[i] as outfile:
            json.dump(data, outfile)
    return 0

init_data_catcher()
print('parte 2 @@@@@@@@@@@@@@@@')
iterator = 0
for t in ticker_list: 
    
    print(t)
    t_0 = datetime.now()

    finance = yf(t)
    summary = finance.get_summary_data()

    now = datetime.now()
    summary = finance.get_summary_data()
    curr_price = finance.get_current_price()
    curr_volume = finance.get_current_volume()
    delta_price = finance.get_current_change()
    curr_bid = summary[t]['bid']
    curr_ask = summary[t]['ask']    
    
    now = datetime.now()
    time_data = [now.hour, now.min, now.second, now.microsecond]
    
    iterator += 1
    t_1 = datetime.now()

    t_sec = t_1.second - t_0.second
    t_net = t_1- t_0

    #sleep(5 - t_sec)

    print(t_net)    

