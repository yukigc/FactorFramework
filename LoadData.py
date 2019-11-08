
#%%
import numpy as np
import pandas as pd
import datetime
import math
import copy

# %%
def get_df(variable, tname, initial_data, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day):
    '''
    tansform series to dataframe
    variable: get a specific variable, like close, totalshare...
    tname: column name of date in initial data, 'TRADE_DT'
    timeindex_day, stock_list: index & columns
    stocks_locate_cum: locate by sorted order
    '''
    df = pd.DataFrame(index = timeindex_day, columns = stock_list)
    df = df.loc[start_day:end_day, :]
    def stock_match(variable,tname, initial_data, s, stocks_locate_cum, timeindex):
        temp = initial_data.loc[stocks_locate_cum.loc[s.name,'start']:stocks_locate_cum.loc[s.name,'end'], [tname,variable]]
        temp = temp.drop_duplicates(tname)
        temp_df = pd.DataFrame(data = temp[variable].values, index = temp[tname], columns = [variable])
        result = pd.DataFrame(index = timeindex, columns = [variable])
        result[variable] = temp_df[variable] 
        return result[variable].values

    df = df.apply(lambda s: stock_match(variable,tname, initial_data, s, stocks_locate_cum, df.index))
    return df
    


def get_locate_cum(initial_data):
    '''
    locate by sorted order
    '''
    stocks_locate = initial_data[['TRADE_DT','S_INFO_WINDCODE']].groupby(by=['S_INFO_WINDCODE']).count()
    stocks_locate_cum = stocks_locate.cumsum()

    stocks_locate_cum['start'] = np.nan
    stocks_locate_cum['end'] = np.nan
    stocks_locate_cum['start'] = stocks_locate_cum['TRADE_DT'].shift(1)
    stocks_locate_cum.loc['000001.SZ', 'start'] = 0
    stocks_locate_cum['start'] = stocks_locate_cum['start'].astype(int)
    stocks_locate_cum['end'] = stocks_locate_cum['TRADE_DT']-1
    return stocks_locate_cum
