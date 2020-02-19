f_in = open("../raw_data_base/stock_tickers_full.csv", 'r').readlines()
f_out = open("croped.txt", 'w')

new_ticker = []

for ticker in f_in:
    f_out.writelines(str(ticker.split('.')[0] + '\n'))
