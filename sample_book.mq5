class CMarketBook;

void OnTick() {

  //create a array
  MqlRates PriceInformation[];
  MqlTick Latest_Price; // Structure to get the latest prices  
  // MqlBookInfo Book_info;

  SymbolInfoTick(Symbol(), Latest_Price);

  static double dBid_price;
  static double dAsk_price;
  static datetime dTime;

  dBid_price = Latest_Price.bid;
  dAsk_price = Latest_Price.ask;
  dTime = Latest_Price.time;

  //sort it from current candle to oldes candles
  ArraySetAsSeries(PriceInformation, true);

  MqlBookInfo priceArray[];
  bool getBook = MarketBookGet(NULL, priceArray);
  if (getBook) {
    int size = ArraySize(priceArray);
    Print("MarketBookInfo about ", Symbol());
  } else {
    Print("Failed to receive DOM for the symbol ", Symbol());
  }

  //copy price data into the array
  int Data = CopyRates(Symbol(), Period(), 0, Bars(Symbol(), Period()), PriceInformation);
  Print("bid price : ", dBid_price, "  ask_price : ", dAsk_price, "  date : ", dTime);
  int size = ArraySize(priceArray);
  for (int i = 0; i < size - 1; i++) {
    Print("type :  ", priceArray[i].type, " price : ", priceArray[i].price, " volume : ", priceArray[i].volume);
  }

}
