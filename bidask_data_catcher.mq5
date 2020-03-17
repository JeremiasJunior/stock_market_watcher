class CMarketBook;

void OnTick() {

  //create a array
  MqlRates PriceInformation[];
  MqlTick Latest_Price; // Structure to get the latest prices  
  // MqlBookInfo Book_info;

  SymbolInfoTick(Symbol(), Latest_Price);

  static double dBid_price;
  static double dAsk_price;
  static int dVolume;
  static datetime dTime;

  dBid_price = Latest_Price.bid;
  dAsk_price = Latest_Price.ask;
  dVolume = Latest_Price.volume_real;
  dTime = Latest_Price.time;

  //sort it from current candle to oldes candles
  ArraySetAsSeries(PriceInformation, true);

  MqlBookInfo priceArray[];
  
  int file = FileOpen("mglu.csv",FILE_WRITE|FILE_READ|FILE_CSV);
 // if(file == INVALID_HANDLE){
 //  Alert("ERROR OPENING FILE");
 //  return;
 // }

  //copy price data into the array
  int data = CopyRates(Symbol(), Period(), 0, Bars(Symbol(), Period()), PriceInformation);
  string info = StringFormat("%s, %.3f, %.3f, %i", TimeToString(dTime), dBid_price, dAsk_price, dVolume);
  Print(info);
  //string toWrite = StringFormat("bid price : ", dBid_price, "  ask_price : ", dAsk_price, "  date : ", dTime);
  FileSeek(file,0,SEEK_END);
  FileWrite(file, info);
  FileClose(file);

}
