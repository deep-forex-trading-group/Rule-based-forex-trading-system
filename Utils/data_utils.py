import pandas as pd

from pandas.testing import assert_series_equal

def get_duka_data_df(path, symbol):
    """
      dukascopy数据应用时间为GMT时间
    """
    assert (symbol.isupper()), "Argument symbol should be uppercase"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["Gmt time"], format="%d.%m.%Y %H:%M:%S.000")
    # filters the weekends and holidays which have not trasactions.
    df = df[df["Volume"]!=0]
    df["symbol"] = symbol
    df = df[["symbol","datetime", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["symbol","datetime", "open", "high", "low", "close", "volume"]
    return df

def get_fm_rc_data_df(path, drop_balance_info=True):
    """
    get the followme records data
    """
    rc_df = pd.read_csv(path)
    # drop the balance infomation
    if drop_balance_info:
        rc_df = rc_df.dropna()
    rc_df["OpenTime"] = pd.to_datetime(rc_df["OpenTime"])
    rc_df["CloseTime"] = pd.to_datetime(rc_df["CloseTime"])
    assert_series_equal(rc_df["FollowmeLots"], rc_df["BrokerLots"], check_names=False) 
    rc_df = rc_df[['Symbol', 'Type', 'FollowmeLots', 
              'OpenTime', 'CloseTime','OpenPrice', 'ClosePrice', 
              'Profit', 'Pips']]
    rc_df.columns = ['symbol', 'action', 'lots', 
                'open_date', 'close_date','open_price', 'close_price', 
                'profits', 'pips']
    return rc_df