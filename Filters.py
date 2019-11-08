#%%
import os
import copy
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_ipo_date(filepath):
    df_ipo_date = pd.read_excel('./Data/IPO_date.xlsx', dtype=str)
    for i in df_ipo_date.index:
        if df_ipo_date.loc[i,'Symbol'][:2] == '00' or df_ipo_date.loc[i,'Symbol'][:2] == '30':
            df_ipo_date.loc[i,'Symbol'] = df_ipo_date.loc[i,'Symbol'] + '.SZ'
        elif df_ipo_date.loc[i,'Symbol'][:2] == '60':
            df_ipo_date.loc[i,'Symbol'] = df_ipo_date.loc[i,'Symbol'] + '.SH'
        else:
            df_ipo_date.loc[i,'Symbol'] = np.nan

    df_ipo_date = df_ipo_date.dropna()

    df_ipo_date_new = pd.DataFrame(df_ipo_date['ListedDate'].values, index = df_ipo_date['Symbol'].values, columns = ['ListedDate'])
    df_ipo_date_new = df_ipo_date_new.applymap(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d"))
    return df_ipo_date_new


def IPO_6m_filters(df_ipo_date, df_monthly_return):

    ipo_filter = pd.DataFrame(index=df_monthly_return.index, columns = df_monthly_return.columns)

    def ipo_stock(s):
        s_filter = copy.deepcopy(s)
        s_ipo_date = df_ipo_date.loc[s.name, 'ListedDate']
        s_ipo_date_6m = s_ipo_date + datetime.timedelta(6*30)
        s_filter[:s_ipo_date_6m] = False
        s_filter[s_ipo_date_6m:] = True
        return s_filter 

    ipo_filter = ipo_filter.apply(lambda s: ipo_stock(s))
    return ipo_filter



def trading_records_filters(daily_trade_volume, df_monthly_return, mon_ratio = 0.75, year_ratio = 0.5, if_plot_ts = False):
    '''
    Trading records filter: If traded then volume cannot be 0; Most recent month records >= 15 days ; Past 12 months records >= 120 days
    Trading calendar adjustment: filter by ratios, mon_ratio = 0.75, year_ratio = 0.5
    '''
    trade_calendar = daily_trade_volume.groupby(by=[daily_trade_volume.index.year, daily_trade_volume.index.month]).count()
    trade_month_cnt = pd.DataFrame(trade_calendar.iloc[:,0].values, index = df_monthly_return.index, columns = ['total_cnt'])
    #temp = daily_trade_volume[daily_trade_volume!=0].groupby(by = [daily_trade_volume.index.year, daily_trade_volume.index.month]).count()
    temp = daily_trade_volume[daily_trade_volume.notnull()].groupby(by = [daily_trade_volume.index.year, daily_trade_volume.index.month]).count()
    days_cnt_mon = pd.DataFrame(temp.values, index = trade_month_cnt.index, columns = daily_trade_volume.columns)
    
    # past 12 months (cumulative sum in first 12 months)
    trade_12month_cnt = trade_month_cnt.rolling(window=12).sum()
    trade_12month_cnt[:12] = trade_month_cnt.cumsum()[:12]
    days_cnt_12mon = days_cnt_mon.rolling(window=12).sum()
    days_cnt_12mon[:12] = days_cnt_mon.cumsum()[:12]
    
    filter_mon = days_cnt_mon.apply(lambda x: x>trade_month_cnt.iloc[:,0]*mon_ratio)
    filter_12mon = days_cnt_12mon.apply(lambda x: x>trade_12month_cnt.iloc[:,0]*year_ratio)
    filter_result = (filter_mon & filter_12mon)
    
    if if_plot_ts:
        plt.figure(figsize=(8, 4))
        plt.plot(filter_result.sum(axis=1))
        plt.title("Number of stocks in universe after trading records filter")
        plt.show()
        
    return filter_result



def size_filters(df_size, last_filter, bottom_percentile = 0.3, if_plot_ratios = False):
    '''
    30% of bottom cap
    '''
    df_size_cond = df_size[last_filter]
    if if_plot_ratios:
        cap_ratio = pd.DataFrame(index = df_size_cond.index, columns = ['bottom_cap_ratio'])
        cap_ratio['bottom_cap_ratio'] = [df_size_cond.loc[t,:][df_size_cond.loc[t,:]<=df_size_cond.loc[t,:].quantile(bottom_percentile)].sum()/df_size.loc[t,:].sum() for t in df_size.index]
        plt.figure(figsize=(8, 4))
        plt.plot(cap_ratio['bottom_cap_ratio'])
        plt.title("Cap ratio of bottom 30% size stock")
        plt.show()
    
    df_size_filters = df_size_cond.apply(lambda x: x>x.quantile(bottom_percentile) , axis=1)
    return df_size_filters



def value_filters(df_value):
    '''
    eliminate nan
    '''
    return df_value.notnull()



# %%
