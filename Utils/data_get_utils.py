import pandas as pd
import numpy as np

from datetime import timedelta
from datetime import datetime
from pandas.testing import assert_series_equal

def calc_profit(action, open_price, close_price, lots, comission, swaps, decimal_pips):
    pips_change = (close_price - open_price) * decimal_pips if action == "buy" \
              else (open_price - close_price) * decimal_pips
    profits = pips_change * lots + commision + swaps
    return profits

def calc_profit(rc_df, act_cname="action", open_price_cname="open_price", close_price_cname="close_price", 
           lots_cname="lots", comm_cname="commission", swaps_cname="swaps", decimal_pips=10000, 
           output_cname="profits_calc", 
           check_pips_eq=True, check_output_col=True):
    rc_pips = np.where(rc_df[act_cname]=='buy', 
                 (rc_df[close_price_cname]-rc_df[open_price_cname]) * decimal_pips, 
                 (rc_df[open_price_cname]-rc_df[close_price_cname]) * decimal_pips)
    if check_pips_eq:
        np.testing.assert_array_almost_equal(np.array(rc_pips), np.array(rc_df["pips"]), decimal=1, verbose=True)
    rc_profits_calc_itself_col_series = rc_pips*rc_df[lots_cname]*10 + rc_df[comm_cname] + rc_df[swaps_cname]
    if check_output_col:
        np.testing.assert_array_almost_equal(np.array(rc_profits_calc_itself_col_series), np.array(rc_df[output_cname]), 
                              decimal=1, verbose=True)
    if check_output_col:
        rc_df[output_cname] = rc_profits_calc_itself_col_series
    return rc_profits_calc_itself_col_series


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

def get_fm_rc_data_df(path, drop_balance_info=True, time_delta_hour=3, decimal_pips=10000):
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
    # filter the meaningless with 0 lots record, weekends and holidays
    rc_df = rc_df[rc_df["lots"] != 0]
    
    # caculate the profit based on the raw price variation data
    rc_df["profits_calc"] = (rc_df["pips"]/decimal_pips)*rc_df["lots"]*decimal_pips + rc_df["commission"] + rc_df["swaps"]
    
    # calculate the commision counted as the pips change
    rc_df["commission_by_pips"] = rc_df["commission"]/rc_df["lots"]/10
    
    # calculate the date variation for calculating swaps
    def calc_date_change(row):
        open_date = row["close_date"].to_pydatetime().replace(hour=0, minute=0, second=0)
        close_date = row["open_date"].to_pydatetime().replace(hour=0, minute=0, second=0)
        
        return (open_date - close_date).days
        
    rc_df["date_var"] = rc_df[["open_date", "close_date"]].apply(calc_date_change, axis=1)
    
    # calculate the swaps by time variations and pips
    rc_df["swap_per_date_var_and_pip"] = np.where(rc_df["date_var"] != 0, rc_df["swaps"]/rc_df["date_var"]/rc_df["lots"]/10, 0)
    
    return rc_df