import pandas as pd

def get_duka_df(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["Gmt time"], format="%d.%m.%Y %H:%M:%S.000")
    df = df[["datetime", "Open", "High", "Low", "Close", "Volume"]]
    return df