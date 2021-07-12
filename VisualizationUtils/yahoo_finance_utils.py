from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np

def data_converter(price_data, code, asset):
    if asset == 'FX':
        code = str(code[3:] if code[:3]=='USD' else code) + "=X"
        
    columns = ["date", "open", "close", "low", "high"]
    price_dict = price_data[code]["prices"]
    date = [p['formatted_date'] for p in price_dict]
    price = [ [np.around(p[c],4) for c in columns] for p in price_dict ]
    
    data = pd.DataFrame(price, columns=columns)
    data["date"] = pd.Series(date, index=data.index)
    return data

def get_yahoo_fx_df(start_date, end_date, curr):
    FXObj = YahooFinancials([(curr+"=X")])
    Fx_Daily = FXObj.get_historical_price_data(start_date, end_date, 'daily')
    df = data_converter(Fx_Daily, curr, 'FX')
    return df