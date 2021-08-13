from product_info import *
from scipy.stats import kurtosis
from scipy.stats import skew
## path of our program
#HEAD_PATH = "/mnt/hgfs/intern"
HEAD_PATH = "d:/intern"


## path of data
DATA_PATH = HEAD_PATH + "/pkl tick/"

## path of the day-to-night data set
NIGHT_PATH = HEAD_PATH + "/night pkl tick/"

## get all the dates of product
import os
def get_dates(product):
    return list(map(lambda x: x[:8] ,os.listdir(DATA_PATH + product)))

## get the data of product of specific date
import pandas as pd
import _pickle as cPickle
import gzip
#import lz4.frame
def get_data(product, date):
    data = load(DATA_PATH + product+"/"+date+".pkl")
    return data


def load(path):
    with gzip.open(path, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    return cPickle.loads(raw_data)

def save(data, path):
    serialized = cPickle.dumps(data)
    with gzip.open(path, 'wb', compresslevel=1) as file_object:
        file_object.write(serialized)

        
import dask
from dask import compute, delayed
def parLapply(CORE_NUM, iterable, func, *args, **kwargs):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute([delayed(f_par)(item) for item in iterable])[0]
        return result
        
        
## returns 0 if the numerator or denominator is 0
import numpy as np
import warnings
def zero_divide(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.divide(x,y)
    if hasattr(y, "__len__"):
        res[y == 0] = 0
    elif y == 0:
        res = 0
        
    return res

def ewma(x, halflife, init=0, adjust=False):
    init_s = pd.Series(data=init)
    s = init_s.append(x)
    if adjust:
        xx = range(len(x))
        lamb=1 - 0.5**(1 / halflife)
        aa=1-np.power(1-lamb, xx)*(1-lamb)
        bb=s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]
        return bb/aa
    else:
        return s.ewm(halflife=halflife, adjust=False).mean().iloc[1:]

def ewma_lambda(x, lambda_, init=0, adjust=False):
    init_s = pd.Series(data=init)
    s = init_s.append(x)
    if adjust:
        xx = range(len(x))
        aa=1-np.power(1-lambda_, xx)*(1-lambda_)
        bb=s.ewm(alpha=lambda_, adjust=False).mean().iloc[1:]
        return bb/aa
    else:
        return s.ewm(alpha=lambda_, adjust=False).mean()[1:]

## moving sum of x
## we don't use rollSum because rollSum would make the first n data to be zero
def cum(x, n):
    sum_x = x.cumsum()
    sum_x_shift = sum_x.shift(n)
    sum_x_shift[:n]= 0
    return sum_x - sum_x_shift


def sharpe(x):
    return zero_divide(np.mean(x)* np.sqrt(250), np.std(x, ddof=1))

def drawdown(x):
    y = np.cumsum(x)
    return np.max(y)-np.max(y[-1:])

def max_drawdown(x):
    y = np.cumsum(x)
    return np.max(np.maximum.accumulate(y)-y)

from collections import OrderedDict
def get_hft_summary(result, thre_mat, n):
    all_result = pd.DataFrame(data={"daily.result": result})
    daily_num = all_result['daily.result'].apply(lambda x: x["num"])
    daily_pnl = all_result['daily.result'].apply(lambda x: x["pnl"])
    total_num = daily_num.sum()
    if len(total_num) != len(thre_mat):
        raise selfException("Mismatch!")
    total_pnl = daily_pnl.sum()
    avg_pnl = zero_divide(total_pnl, total_num)

    total_sharp = sharpe(daily_pnl)
    total_drawdown = drawdown(daily_pnl)
    total_max_drawdown = max_drawdown(daily_pnl)

    final_result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"]), ("close", thre_mat["close"]), ("num", total_num),
                                                 ("avg.pnl", avg_pnl), ("total.pnl", total_pnl), ("sharp", total_sharp), 
                                                 ("drawdown", total_drawdown), ("max.drawdown", total_max_drawdown), 
                                                 ("mar", total_pnl/total_max_drawdown)]), 
                                index=thre_mat.index)
    
    return OrderedDict([("final.result", final_result), ("daily.num", daily_num), ("daily.pnl", daily_pnl)])


from collections import OrderedDict
def get_signal_pnl(file, product, signal_name, thre_mat, reverse=1, tranct=1.1e-4, max_spread=0.61, tranct_ratio=True, HEAD_PATH="d:/intern"):
    ## load data
    data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file)
    S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
    pred = S*reverse
    pred = pred[data["good"]]
    data = data[data["good"]].reset_index(drop=True)
    #n_bar = len(data)
    
    ## load signal
    
    ## we don't know the signal is positive correlated or negative correlated  
    #n_thre = len(thre_mat)
    result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"].values), ("close", thre_mat["close"].values),
                               ("num", 0), ("avg.pnl", 0), ("pnl", 0)]), index=thre_mat.index)
    count = 0;
    cur_spread = data["ask"]-data["bid"]
    for thre in thre_mat.iterrows():
        count = count+1
        buy = pred>thre[1]["open"]
        sell = pred<-thre[1]["open"]
        signal = pd.Series(data=0, index=data.index)
        position = signal.copy()
        signal[buy] = 1
        signal[sell] = -1
        scratch = -thre[1]["close"]
        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[(signal==1) & (data["next.ask"]>0) & (data["next.bid"]>0) & (cur_spread<max_spread) & (data["can_trade"])] = 1
        position_pos[(pred< -scratch) & (data["next.bid"]>0) & (cur_spread<max_spread) & data["can_trade"]] = 0
        position_pos.ffill(inplace=True)
        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[(signal==-1) & (data["next.ask"]>0) & (data["next.bid"]>0) & (cur_spread<max_spread) & (data["can_trade"])] = -1
        position_neg[(pred> scratch) & (data["next.ask"]>0) & (cur_spread<max_spread) & (data["can_trade"])] = 0
        position_neg.ffill(inplace=True)
        position = position_pos + position_neg
        #position[n_bar-1] = 0
        position.iloc[0] = 0
        position.iloc[-2:] = 0
        change_pos = position - position.shift(1)
        change_pos.iloc[0] = 0
        change_base = pd.Series(data=0, index=data.index)
        change_buy = change_pos>0
        change_sell = change_pos<0
        if (tranct_ratio):
            change_base[change_buy] = data["next.ask"][change_buy]*(1+tranct)
            change_base[change_sell] = data["next.bid"][change_sell]*(1-tranct)
        else:
            change_base[change_buy] = data["next.ask"][change_buy]+tranct
            change_base[change_sell] = data["next.bid"][change_sell]-tranct
        final_pnl = -sum(change_base*change_pos)
        num = sum((position!=0) & (change_pos!=0))
        if num == 0:
            avg_pnl = 0
            final_pnl = 0
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            return result
        else:
            avg_pnl = np.divide(final_pnl, num)
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
    return result

def vanish_thre(x, thre):
    x[np.abs(x)>thre] = 0
    return x


import itertools
def create_signal_path(signal_list, product, HEAD_PATH):
    keys = list(signal_list.params.keys())
    for cartesian in itertools.product(*signal_list.params.values()):
        signal_name = signal_list.factor_name
        for i in range(len(cartesian)):
            signal_name = signal_name.replace(keys[i], str(cartesian[i]))
        
        os.makedirs(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name, exist_ok=True)
        print(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name)
        
def get_sample_signal(good_night_files, sample, product, signal_list, period, daily_num):
    n_samples = sum(daily_num[sample])
    n_days = sum(sample)
    n_signal = len(signal_list)
    all_signal =  np.ndarray(shape=(int(n_samples),n_signal))
    cur = 0
    for file in good_night_files[sample]:
        data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file)
        chosen = (np.arange(sum(data["good"]))+1) % period==0
        n_chosen = sum(chosen)
        for i in range(n_signal):
            signal_name = signal_list[i]
            S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
            S = S[data["good"]]
            signal = S[(np.arange(len(S))+1) % period == 0]
            signal[np.isnan(signal)] = 0 ## the ret.cor has some bad records
            signal[np.isinf(signal)] = 0 ## the ret.cor has some bad records
            all_signal[cur:(cur+n_chosen),i] = signal
        cur = cur+n_chosen
    all_signal = pd.DataFrame(all_signal, columns=signal_list)
    return all_signal

def get_signal_pnl_better(file, product, signal_name, thre_mat, reverse=1, tranct=1.1e-4, min_spread=1.1, HEAD_PATH="d:/intern/python"):
    ## load data
    data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file)
    data = data[data["good"]]
    #n_bar = len(data)
    
    ## load signal
    S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
    
    ## we don't know the signal is positive correlated or negative correlated  
    pred = S*reverse
    #n_thre = len(thre_mat)
    result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"].values), ("close", thre_mat["close"].values),
                               ("num", 0), ("avg.pnl", 0), ("pnl", 0)]), index=thre_mat.index)
    
    bid_ask_spread = data["ask"]-data["bid"]
    next_spread = bid_ask_spread.shift(-1)
    next_spread.iloc[-1] = bid_ask_spread.iloc[-1]
    not_trade = (data["time"]=="10:15:00") | (data["time"]=="11:30:00") | (data["time"]=="15:00:00") | (bid_ask_spread>min_spread) | (next_spread>min_spread)

    for thre in thre_mat.iterrows():
        buy = pred>thre[1]["open"]
        sell = pred<-thre[1]["open"]
        signal = pd.Series(data=0, index=data.index)
        position = signal.copy()
        signal[buy] = 1
        signal[sell] = -1
        signal[not_trade] = 0
        scratch = -thre[1]["close"]
        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[(signal==1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = 1
        position_pos[(pred< -scratch) & (data["next.bid"]>0)] = 0
        position_pos.ffill(inplace=True)
        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[(signal==-1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = -1
        position_neg[(pred> scratch) & (data["next.ask"]>0)] = 0
        position_neg.ffill(inplace=True)
        position = position_pos + position_neg
        #position[n_bar-1] = 0
        position.iloc[0] = 0
        position.iloc[-2:] = 0
        change_pos = position - position.shift(1)
        change_pos.iloc[0] = 0
        change_base = pd.Series(data=0, index=data.index)
        change_buy = change_pos>0
        change_sell = change_pos<0
        change_base[change_buy] = (data["next.ask"][change_buy])*(1+tranct)
        change_base[change_sell] = (data["next.bid"][change_sell])*(1-tranct)
        final_pnl = -sum(change_base*change_pos)

        num = sum((position!=0) & (change_pos!=0))
        
        if num == 0:
            avg_pnl = 0
            final_pnl = 0
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            return result
        else:
            avg_pnl = np.divide(final_pnl, num)
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            
    return result




def get_range_pos(wpr, min_period, max_period, period):
    return ewma(zero_divide(wpr-min_period, max_period-min_period), period, adjust=True) - 0.5


import dask
from dask import compute, delayed
import matplotlib.pyplot as plt
import functools

from collections import OrderedDict
def get_signal_stat(signal_name, thre_mat, product, good_night_files, split_str="2018", reverse=1, tranct=1.1e-4, 
                    max_spread=0.61, tranct_ratio=True, min_pnl=2, min_num=20):
    train_sample = good_night_files<split_str
    test_sample = good_night_files>split_str
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse,
                                 tranct=tranct, max_spread=max_spread, tranct_ratio=tranct_ratio)
        train_result = compute([delayed(f_par)(file) for file in good_night_files[train_sample]])[0]
    train_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
    good_strat = (train_stat["final.result"]["avg.pnl"]>min_pnl) & (train_stat["final.result"]["num"]>min_num)
    print("good strategies: \n", good_strat[good_strat], "\n")
    
    good_pnl = train_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    
    print("train sharpe: ", sharpe(good_pnl), "\n")
    
    date_str = [n[0:8] for n in good_night_files]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    plt.figure(1, figsize=(16, 10))
    plt.title("train")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[train_sample], good_pnl.cumsum())
    
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse,
                                 tranct=tranct, max_spread=max_spread, tranct_ratio=tranct_ratio)
        test_result = compute([delayed(f_par)(file) for file in good_night_files[test_sample]])[0]
    test_stat = get_hft_summary(test_result, thre_mat, sum(test_sample))
  
    test_pnl = test_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
   
    print("test sharpe: ", sharpe(test_pnl), "\n")
    
    plt.figure(2, figsize=(16, 10))
    plt.title("test")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[test_sample], test_pnl.cumsum())
    return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat), ("good.strat", good_strat)])

from collections import OrderedDict
def get_signal_stat_better(signal_name, thre_mat, product, good_night_files, split_str="2018", reverse=1, min_pnl=2, CORE_NUM=4):
    train_sample = good_night_files<split_str
    test_sample = good_night_files>split_str
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_better, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse)
        train_result = compute([delayed(f_par)(file) for file in good_night_files[train_sample]])[0]
    train_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
    good_strat = train_stat["final.result"]["avg.pnl"]>min_pnl
    print("good strategies: \n", good_strat[good_strat], "\n")
    
    good_pnl = train_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    
    print("train sharpe: ", sharpe(good_pnl), "\n")
    
    date_str = [n[0:8] for n in good_night_files]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    plt.figure(1, figsize=(16, 10))
    plt.title("train")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[train_sample], good_pnl.cumsum())
    
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_better, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse)
        test_result = compute([delayed(f_par)(file) for file in good_night_files[test_sample]])[0]
    test_stat = get_hft_summary(test_result, thre_mat, sum(test_sample))
    
    test_pnl = test_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    
    print("test sharpe: ", sharpe(test_pnl), "\n")
    
    plt.figure(2, figsize=(16, 10))
    plt.title("test")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[test_sample], test_pnl.cumsum())
    
    return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat), ("good.strat", good_strat)])



def rsi(ret, period):
    abs_move = np.abs(ret)
    up_move = np.maximum(ret, 0)
    up_total = ewma(up_move, period, adjust=True)
    move_total = ewma(abs_move, period, adjust=True)
    rsi = zero_divide(up_total, move_total) - 0.5
    return rsi


def get_all_signal(good_night_files, product, signal_name, period):
    n_good_night = len(good_night_files)
    to_choose = (np.arange(n_good_night)+1) % 10 == 0
    all_signal = np.array([])
    for file in good_night_files[to_choose]:
        S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
        data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file)
        signal = S[data["good"]]
        chosen = (np.arange(len(signal))+1) % period==0
        all_signal = np.concatenate((all_signal, signal[chosen]), axis=0)
    return all_signal


def add_min_max(file, period_list):
    data = load(file)
    for period in period_list:
        data["min."+str(period)] = data["wpr"].rolling(period).min()
        #data.loc[:period-1, ("min."+str(period))] = np.minimum.accumulate(data["wpr"].iloc[:period-1])
        data.loc[:period-1, ("min."+str(period))] = data["wpr"][0]
        
        data["max."+str(period)] = data["wpr"].rolling(period).max()
        #data.loc[:period-1, ("max."+str(period))] = np.maximum.accumulate(data["wpr"].iloc[:period-1])
        data.loc[:period-1, ("max."+str(period))] = data["wpr"][0]
    
    save(data, file)
    #print("end ", file)
    
    
def get_signal_pnl_close(file, product, signal_name, thre_mat, reverse=1, rebate=0, HEAD_PATH="d:/intern"):
    ## load data
    data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file)
    ## load signal
    S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file)
    ## we don't know the signal is positive correlated or negative correlated  
    pred = S*reverse
    pred = pred[data["good"]]
    data = data[data["good"]]
    

    ## load product info
    tranct = product_info[product]["tranct"]*(1-rebate)
    min_spread = product_info[product]["spread"]+0.001
    close = product_info[product]["close"]*(1-rebate)
    tranct_ratio = product_info[product]["tranct.ratio"]
    
    
    result = pd.DataFrame(data=OrderedDict([("open", thre_mat["open"].values), ("close", thre_mat["close"].values),
                               ("num", 0), ("avg.pnl", 0), ("pnl", 0)]), index=thre_mat.index)
    
    bid_ask_spread = data["ask"]-data["bid"]
    next_spread = bid_ask_spread.shift(-1)
    next_spread.iloc[-1] = bid_ask_spread.iloc[-1]
    not_trade = (data["time"]=="10:15:00") | (data["time"]=="11:30:00") | (data["time"]=="15:00:00") | (bid_ask_spread>min_spread) | (next_spread>min_spread)

    for thre in thre_mat.iterrows():
        buy = pred>thre[1]["open"]
        sell = pred<-thre[1]["open"]
        signal = pd.Series(data=0, index=data.index)
        position = signal.copy()
        signal[buy] = 1
        signal[sell] = -1
        signal[not_trade] = 0
        scratch = -thre[1]["close"]
        position_pos = pd.Series(data=np.nan, index=data.index)
        position_pos.iloc[0] = 0
        position_pos[(signal==1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = 1
        position_pos[(pred< -scratch) & (data["next.bid"]>0)] = 0
        position_pos.ffill(inplace=True)
        position_neg = pd.Series(data=np.nan, index=data.index)
        position_neg.iloc[0] = 0
        position_neg[(signal==-1) & (data["next.ask"]>0) & (data["next.bid"]>0)] = -1
        position_neg[(pred> scratch) & (data["next.ask"]>0)] = 0
        position_neg.ffill(inplace=True)
        position = position_pos + position_neg
        position.iloc[0] = 0
        position.iloc[-2:] = 0
        change_pos = position - position.shift(1)
        change_pos.iloc[0] = 0
        #change_base = pd.Series(data=0, index=data.index)
        
        pre_pos = position.shift(1)
        pre_pos.iloc[0] = 0
        open_buy = (pre_pos<=0) & (position>0)
        open_sell = (pre_pos>=0) & (position<0)
        close_buy = (pre_pos<0) & (position>=0)
        close_sell = (pre_pos>0) & (position<=0)
        open_buy_pnl = pd.Series(data=0, index=data.index)
        open_sell_pnl = pd.Series(data=0, index=data.index)
        close_buy_pnl = pd.Series(data=0, index=data.index)
        close_sell_pnl = pd.Series(data=0, index=data.index)
        
        if tranct_ratio:
            open_buy_pnl[open_buy] = -data["next.ask"][open_buy]*(1+tranct)
            open_sell_pnl[open_sell] = data["next.bid"][open_sell]*(1-tranct)
            close_buy_pnl[close_buy] = -data["next.ask"][close_buy]*(1+close)
            close_sell_pnl[close_sell] = data["next.bid"][close_sell]*(1-close)
        else:
            open_buy_pnl[open_buy] = -data["next.ask"][open_buy]-tranct
            open_sell_pnl[open_sell] = data["next.bid"][open_sell]-tranct
            close_buy_pnl[close_buy] = -data["next.ask"][close_buy]-close
            close_sell_pnl[close_sell] = data["next.bid"][close_sell]-close
        
        final_pnl = sum(open_buy_pnl+open_sell_pnl+close_buy_pnl+close_sell_pnl)
        
        num = sum((position!=0) & (change_pos!=0))
        
        if num == 0:
            avg_pnl = 0
            final_pnl = 0
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            return result
        else:
            avg_pnl = np.divide(final_pnl, num)
            result.loc[thre[0], ("num", "avg.pnl", "pnl")] = (num, avg_pnl, final_pnl)
            
    return result

def fast_roll_var(x, period):
      x_ma = cum(x,period)/period
      x2 = x*x
      x2_ma = cum(x2,period)/period
      var_x = x2_ma-x_ma*x_ma
      return(var_x)

def fast_roll_cor_ewma(x, y, period):
    x_ma = ewma(x,period)
    x2 = x*x
    x2_ma = ewma(x2,period)
    var_x = x2_ma-x_ma*x_ma
    var_x[var_x<0] = 0
    y_ma = ewma(y,period)
    y2 = y*y
    y2_ma = ewma(y2,period)
    var_y = y2_ma-y_ma*y_ma
    var_y[var_y<0] = 0
    upper = ewma(x*y, period) - x_ma*y_ma
    result = zero_divide(upper, np.sqrt(var_x*var_y))
    return (result)


def fast_roll_cor(x, y, period):
    x_ma = cum(x,period)/period
    x2 = x*x
    x2_ma = cum(x2,period)/period
    var_x = x2_ma-x_ma*x_ma
    var_x[var_x<0] = 0
    y_ma = cum(y,period)/period
    y2 = y*y
    y2_ma = cum(y2,period)/period
    var_y = y2_ma-y_ma*y_ma
    var_y[var_y<0] = 0
    #upper = (x-x_ma)*(y-y_ma)
    #result = zero_divide(cum(upper, period), np.sqrt(var_x*var_y))/period
    upper = cum(x*y, period) - period*x_ma*y_ma
    result = zero_divide(upper, np.sqrt(var_x*var_y))/period
    return (result)

def fast_ret_cor(ret, period):
    pre_ret = ret.shift(1)
    pre_ret[0] = 0
    rolling_cor = fast_roll_cor(pre_ret, ret, period)
    rolling_cor.iloc[:period-1] = 0
    return rolling_cor

def fast_ret_cor_ewma2(ret, period):
    pre_ret = ret.shift(1)
    pre_ret[0] = 0
    result = fast_roll_cor_ewma(pre_ret, ret, period)*ewma(ret,period)*period
    result.iloc[:period-1]=0
    return result

def fast_ret_cor_ewma(ret, period):
    pre_ret = ret.shift(1)
    pre_ret[0] = 0
    result = fast_roll_cor_ewma(pre_ret, ret, period)*ewma(ret,period)*period
    result = np.asarray(result)
    return result

def vol_cor(ret, qty, period):
    result = fast_roll_cor_ewma(qty, ret, period)*ewma(np.abs(ret),period)*period
    return result

def check_strat_prob(train_pnl, test_pnl, num=10000):
    random.seed([100])
    aa = np.random.standard_normal(num).reshape(-1,num)
    aa.sum(axis=1)
    
def fcum(x, n, fill=0):
    return pd.Series(data=cum(pd.concat((x, pd.Series(np.repeat(fill, n))), ignore_index=True), n).shift(-n)[:-n].values, index=x.index)

    
def get_signal_vec(signal_mat, signal_name, product, file_name):
    S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file_name)
    S[np.isnan(S)] = 0
    if signal_mat is None:
        return S
    else:
        return np.vstack((signal_mat, S))

def get_signal_mat(file_name, product, signal_list, HEAD_PATH):
    signal_mat = functools.reduce(functools.partial(get_signal_vec, product=product, file_name=file_name), signal_list, None)
    save(signal_mat, HEAD_PATH+"/signal mat pkl/"+product+"/"+file_name)
    
def get_daily_pred(file_name, product, signal_list, coef, strat, HEAD_PATH):
    signal_mat = load(HEAD_PATH+"/signal mat pkl/"+product+"/"+file_name)
    if len(coef)>1:
        S = np.dot(signal_mat.T, coef)
    else:
        S = signal_mat * coef
    save(S, HEAD_PATH+"/tmp pkl/"+product+"/"+strat+"/"+file_name)

from scipy.optimize import minimize
def TotalTRC(x, Cov):
    x = np.append(x, 1-np.sum(x))
    TRC = np.prod((np.dot(Cov, x), x), axis=0)
    if np.sum(x<0)>0: 
        return 10**12
    else:
        return np.sum((TRC[:, None] - TRC) ** 2)

def risk_parity(Sub, maxiter=9999):
    m = Sub.shape[1]
    Cov = np.cov(Sub, rowvar=False)
    res = minimize(functools.partial(TotalTRC, Cov=Cov), np.repeat(1/m, m-1), method="Nelder-Mead", options={'xtol': 1e-6, "maxiter": maxiter, "disp":True})
    w = np.append(res["x"], 1-np.sum(res["x"]))
    #res = nelder_mead(functools.partial(TotalTRC, Cov=Cov), np.repeat(1/m, m-1), step=1e-3, no_improve_thr=1e-05)
    #w = np.append(res[0], 1-np.sum(res[0]))
    return w
    

def get_signal_stat_close(signal_name, thre_mat, product, good_night_files, split_str="2018", reverse=1, min_pnl=2, min_num=10, rebate=0, CORE_NUM=4):
    train_sample = good_night_files<split_str
    test_sample = good_night_files>split_str
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_close, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse, rebate=rebate)
        train_result = compute([delayed(f_par)(file) for file in good_night_files[train_sample]])[0]
    train_stat = get_hft_summary(train_result, thre_mat, sum(train_sample))
    good_strat = (train_stat["final.result"]["avg.pnl"]>=min_pnl) & (train_stat["final.result"]["num"]>=min_num)
    if sum(good_strat)==0:
        print("no good strategy!")
        return 0
    print("good strategies: \n", good_strat[good_strat], "\n")
    good_pnl = train_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    print("train sharpe: ", sharpe(good_pnl), "\n")
    date_str = [n[0:8] for n in good_night_files]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    plt.figure(1, figsize=(16, 10))
    plt.title("train")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[train_sample], good_pnl.cumsum())
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_close, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse, rebate=rebate)
        test_result = compute([delayed(f_par)(file) for file in good_night_files[test_sample]])[0]
    test_stat = get_hft_summary(test_result, thre_mat, sum(test_sample))
    test_pnl = test_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    print("test sharpe: ", sharpe(test_pnl), "\n")
    plt.figure(2, figsize=(16, 10))
    plt.title("test")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[test_sample], test_pnl.cumsum())
    return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat), ("good.strat", good_strat)])    


def get_signal_stat_roll(signal_name, thre_mat, product, good_night_files, train_sample, test_sample,
                         reverse=1, min_pnl=2, min_num=10, CORE_NUM=4):
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_close, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse)
        train_result = compute([delayed(f_par)(file) for file in good_night_files[train_sample]])[0]
    train_stat = get_hft_summary(train_result, thre_mat, len(train_sample))
    good_strat = (train_stat["final.result"]["avg.pnl"]>=min_pnl) & (train_stat["final.result"]["num"]>=min_num)
    if sum(good_strat)==0:
        print("no good strategy!")
        return 0
    print("good strategies: \n", good_strat[good_strat], "\n")
    good_pnl = train_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    print("train sharpe: ", sharpe(good_pnl), "\n")
    date_str = [n[0:8] for n in good_night_files]
    format_dates = np.array([pd.to_datetime(d) for d in date_str])
    plt.figure(1, figsize=(16, 10))
    plt.title("train")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[train_sample], good_pnl.cumsum())
    with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
        f_par = functools.partial(get_signal_pnl_close, product=product, signal_name=signal_name, thre_mat=thre_mat, reverse=reverse)
        test_result = compute([delayed(f_par)(file) for file in good_night_files[test_sample]])[0]
    test_stat = get_hft_summary(test_result, thre_mat, len(test_sample))
    test_pnl = test_stat["daily.pnl"].loc[:, good_strat].sum(axis=1)/sum(good_strat)
    print("test sharpe: ", sharpe(test_pnl), "\n")
    plt.figure(2, figsize=(16, 10))
    plt.title("test")
    plt.xlabel("date")
    plt.ylabel("pnl")
    plt.plot(format_dates[test_sample], test_pnl.cumsum())
    return OrderedDict([("train.stat", train_stat), ("test.stat", test_stat), ("good.strat", good_strat)])    


def evaluate_signal(signal, good_night_files, product, min_pnl, min_num, HEAD_PATH, period=4096):
    all_signal = get_all_signal(good_night_files, product, signal+"."+ str(period), period)
    tranct = product_info[product]["tranct"]
    tranct_ratio = product_info[product]["tranct.ratio"]
    open_thre = np.quantile(abs(all_signal), np.append(np.arange(0.991, 0.999, 0.001),
                                                  np.arange(0.9991,0.9999,0.0001)))
    thre_mat = pd.DataFrame(data=OrderedDict([("open", open_thre), ("close", -open_thre)]))
    print("reverse=1")
    signal_stat = get_signal_stat_close(signal+"."+str(period), thre_mat, product, good_night_files,
                                   reverse=1, min_pnl = min_pnl, min_num=min_num)
    print("reverse=-1")
    signal_stat = get_signal_stat_close(signal+"."+str(period), thre_mat, product, good_night_files,
                                 reverse=-1, min_pnl = min_pnl, min_num=min_num)


    
def get_daily_gbm(file_name, product, signal_list, model, strat, HEAD_PATH, thre=float('Inf')):
    data = load(HEAD_PATH+"/night pkl tick/"+product+"/"+file_name)
    def signal_mat_reduce(x, signal_name):
        S = load(HEAD_PATH+"/tmp pkl/"+product+"/"+signal_name+"/"+file_name)
        S[np.isnan(S)] = 0
        S[np.isinf(S)] = 0
        S = np.asarray(S)
        if x is None:
            x = pd.DataFrame(S)
        else:
            x = pd.concat((x, pd.Series(S)), axis=1)
        return x
    signal_mat = functools.reduce(signal_mat_reduce, signal_list, None)
    signal_mat.columns = signal_list
    S = model.predict(signal_mat)
    S[np.abs(S)>thre] = 0
    save(S, HEAD_PATH+"/tmp pkl/"+product+"/"+strat+"/"+file_name)

def get_glmnet_ensemble_roll_model(train_start, train_end, forward_len, alpha=1, start_year=2018, period=2048):
    cum_daily_ticks = daily_ticks.cumsum()
    if train_start==0:
        train_tick_start = 0
    else:
        train_tick_start = cum_daily_ticks[train_start-1]+1
    train_tick_end = cum_daily_ticks[train_end]-1
    test_tick_start = train_tick_end+2
    test_tick_end = cum_daily_ticks[train_end+1]
    n_signal = len(signal_list)
    nfold = 10
    model_coef = np.zeros((n_signal, n_mod))
    for i_mod in range(n_mod):
        x_train = train_array[i_mod,:,:n_signal]
        y_train = train_array[i_mod,:,n_signal]
        n_train = x_train.shape[0]
        model = ElasticNetCV(l1_ratio=alpha, n_alphas=100, fit_intercept=False, cv=10, max_iter=1000).fit(x_train, y_train)
        model_coef[:,i_mod] = model.coef_
    coef = np.mean(model_coef, axis=1)
    if alpha==1:
        strat = "lasso.ensemble.roll."+str(start_year)+"."+str(period)
    elif alpha==0:
        strat = "ridge.ensemble.roll."+str(start_year)+"."+str(period)
    else:
        strat = "elastic.ensemble.roll."+str(start_year)+"."+str(period)
    os.makedirs(HEAD_PATH+"/roll model", exist_ok=True)
    os.makedirs(HEAD_PATH+"/roll model/"+product, exist_ok=True)
    save(model_coef, HEAD_PATH+"/roll model/"+product+"/"+strat+"."+str(train_start)+"."+str(train_end)+".pkl")
   