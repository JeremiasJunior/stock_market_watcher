import

ticker_list = stock_cmpr.load_ticker_list('stock_tickers.csv')
now = datetime.now()
f_name = (str(now.year) + '-' + str(now.month)+'-'+str(now.day)+'@'+str(now.hour)+'-'+str(now.minute))
f_dir = 'ticker.json'

