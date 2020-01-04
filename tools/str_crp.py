f_in = open("../data_catcher/stock_tickers.csv", 'r').readlines()
f_out = open("stock_tickers_new.csv", 'w')

new_ticker = []

for ticker in f_in:
    f_out.writelines(str(ticker.split('\n')[0])+'.SA\n')
