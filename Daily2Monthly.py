# %%
import numpy as np
import pandas as pd

#%%

def generate_monthly_index(dailyindex):
    '''
    use the last trading day of each month as monthly time index
    '''
    return [dailyindex[(dailyindex.year==y)&(dailyindex.month==m)][-1] for y in range(dailyindex[0].year, dailyindex[-1].year+1) for m in range(1,13)]


def generate_monthly_df(df_daily):
    '''
    extract month-end's close price or total shares
    '''
    monthly_timeline = generate_monthly_index(df_daily.index)
    return df_daily.loc[monthly_timeline, :]


def cal_monthly_return(df_daily_return):
    '''
    calculate daily return first, and use return's product in each month
    '''
    monthly_timeline = generate_monthly_index(df_daily_return.index)
    daily_return = df_daily_return + 1
    daily_return = daily_return.groupby(by=[df_daily_return.index.year, df_daily_return.index.month]).prod()-1
    df_monthly_return = pd.DataFrame(daily_return.values, index = monthly_timeline, columns = df_daily_return.columns)
    return df_monthly_return

