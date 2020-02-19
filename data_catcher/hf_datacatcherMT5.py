import zmq
import json
import time
import sys
from datetime import datetime
from lib import stock_cmpr
import collections
import concurrent.futures

sys.path.append('../')

def remote_send(socket, data):
    try:
        socket.send_string(data)
        msg = socket.recv_string()
        return (msg)
    except zmq.Again as e:
        print ("Waiting for PUSH from MetaTrader 4..")

# Get zmq context
context = zmq.Context()

# Create REQ Socket
req = reqSocket = context.socket(zmq.REQ)
connect = reqSocket.connect("tcp://localhost:90")

ticker_list = stock_cmpr.load_ticker_list('stock_tickers.csv')
#primeira tentativa, abrindo e fechando arquivos em sequencia

def get_data(ticker):
    flag = str("RATES|"+ticker)
    data = remote_send(reqSocket, flag)

perf_s = time.perf_counter()
tl_size = len(ticker_list)
t_time = float()


for i in range(10):
    for ticker in ticker_list:
        f_o = open(str(ticker), 'a')
        flag = str("RATES|"+ticker)
        data = remote_send(reqSocket, flag)
        print(flag,data)
        f_o.writelines(data + "@")
        f_o.close()

perf_f = time.perf_counter()
print(f'\n\nscript finalizado tempo {round(perf_f-perf_s, 2)} segundos...')


'''
tenho que fazer um esquema pra salvar os dados, tenho duas possibilidades
1- colocar tudo num json 
2- armazenar cada ticker em um csv separado

'''


while(False):
    print(remote_send(reqSocket, "RATES|LTCUSD"))

# bid, ask, buy_volume, sell_volume, tick_volume, real_volume, buy_volume_market, sell_volume_market