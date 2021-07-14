import pandas as pd
import numpy as np

from datetime import timedelta
from datetime import datetime
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

def get_fm_rc_data_df(path, drop_balance_info=True, time_delta_hour=3):
    """
     get the followme records data
    """
    rc_df = pd.read_csv(path)
    # drop the balance infomation
    if drop_balance_info:
        rc_df = rc_df.dropna()
    rc_df["OpenTime"] = pd.to_datetime(rc_df["OpenTime"]) - timedelta(hours=time_delta_hour)
    rc_df["CloseTime"] = pd.to_datetime(rc_df["CloseTime"]) - timedelta(hours=time_delta_hour)
    
    # assert that the lots calculated by Followme is equal to which calculated by Brokers
    assert_series_equal(rc_df["FollowmeLots"], rc_df["BrokerLots"], check_names=False) 
    rc_df = rc_df[['Symbol', 'Type', 'FollowmeLots', 
              'OpenTime', 'CloseTime','OpenPrice', 'ClosePrice', 
              'Profit', 'Pips','Commission','Swaps']]
    # modifies the column names to lower cases 
    rc_df.columns = ['symbol', 'action', 'lots', 
                'open_date', 'close_date','open_price', 'close_price', 
                'profits', 'pips','commission','swaps']
    # filter the meaningless with 0 lots record
    rc_df = rc_df[rc_df["lots"] != 0]
    
    # caculate the profit based on the raw price variation data
    rc_df["profits_calc"] = (rc_df["pips"]/10000)*rc_df["lots"]*100000 + rc_df["commission"] + rc_df["swaps"]
    
    # calculate the commision counted as the pips change
    rc_df["commission_by_pips"] = rc_df["commission"]/rc_df["lots"]/10
    
    # calculate the date variation for calculating swaps
    def calc_date_change(row):
        open_date = row["open_date"].to_pydatetime().day
        close_date = row["close_date"].to_pydatetime().day
        return close_date - open_date
        
    rc_df["date_var"] = rc_df[["open_date", "close_date"]].apply(calc_date_change, axis=1)
    
    # calculate the swaps by time variations and pips
    rc_df["swap_by_time_variations_and_pips"] = np.where(rc_df["date_var"] != 0, rc_df["swaps"]/rc_df["date_var"]/rc_df["lots"]/10, 0)
    
    return rc_df