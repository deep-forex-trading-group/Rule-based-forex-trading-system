from pandas import Series
import numpy as np
import pandas as pd
# create a differenced series
def difference(dataset, interval=1):
    diff = list([np.nan])
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def difference_rate(dataset, interval=1):
    diff = list([np.nan])
    for i in range(interval, len(dataset)):
        value = (dataset[i] - dataset[i - interval])/dataset[i - interval]
        diff.append(value)
    return Series(diff)

def reverse_df(df, is_reindex=True):
    # return df.reindex(index=df.index[::-1])
    return df.iloc[::-1].reset_index(drop=True) if is_reindex else df.iloc[::-1]

def calc_ssr(rt):
    rt = rt.astype(np.float64)
    ssr = np.mean(rt) / np.std(rt) / (-np.sum(rt[rt < 0])) * len(rt)
    return ssr

def calc_sr(rt):
    rt = rt.astype(np.float64)
    sr = np.mean(rt) / np.std(rt) * len(rt)
    return sr
    