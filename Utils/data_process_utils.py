import pandas as pd

from pandas.testing import assert_series_equal

def slice_df_by_timerange(df, col_name, start_dt, end_dt):
    return df[(df[col_name] >= start_dt) & (df[col_name] <= end_dt)]

def get_period_bar_date(df, column_names, period_in_minute):
    """
    Transfers the date to fit in the bar time frame
    For example: "2021-07-09 18:14:50" => "2021-07-09 18:10:00" (the second 5-minute bar after 18:00)
    """
    for column_name in column_names:
        output_column_name = (column_name + "_mod_" + str(period_in_minute))
        df[output_column_name] = df[column_name].apply(lambda dt: dt.replace(minute= int(dt.minute / 5) * 5, second = 0 ))
    return df