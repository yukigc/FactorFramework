
#%%
import sys
import numpy as np
import pandas as pd
import datetime
import math
import copy
import LoadData
import Daily2Monthly

# %%

# back test period
start_day = '1999-01-01'
end_day = '2016-12-31'

# load initial data
Daily_return_with_cap = pd.read_pickle('./Data/Daily_return_with_cap.pkl')
stock_list = Daily_return_with_cap['S_INFO_WINDCODE'].unique()
timeindex_day = Daily_return_with_cap['TRADE_DT'].unique()
timeindex_day.sort()

# %%

stocks_locate_cum = LoadData.get_locate_cum(Daily_return_with_cap)

daily_return = LoadData.get_df('adj_pct_chg', 'TRADE_DT',Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)
daily_close = LoadData.get_df('S_DQ_CLOSE' ,'TRADE_DT', Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)
daily_shares_a = LoadData.get_df('S_SHARE_TOTALA' ,'TRADE_DT', Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)
daily_shares_total = LoadData.get_df('TOT_SHR' ,'TRADE_DT',Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)

#%%
df_monthly_return = Daily2Monthly.cal_monthly_return(daily_return)
df_monthly_shares_a = Daily2Monthly.generate_monthly_df(daily_shares_a)
df_monthly_shares_total = Daily2Monthly.generate_monthly_df(daily_shares_total)


# %%
