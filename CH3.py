
#%%
import sys
import numpy as np
import pandas as pd
import datetime
import math
import copy
import LoadData
import Daily2Monthly
import LoadIncome

# if is not initial, load pkl files directly
is_initial = 0

# %%

# back test period
start_day = '1999-01-01'
end_day = '2016-12-31'

# load price,share data
Daily_return_with_cap = pd.read_pickle('./Data/Daily_return_with_cap.pkl')
stock_list = Daily_return_with_cap['S_INFO_WINDCODE'].unique()
timeindex_day = Daily_return_with_cap['TRADE_DT'].unique()
timeindex_day.sort()

if is_initial:

    stocks_locate_cum = LoadData.get_locate_cum(Daily_return_with_cap, 'TRADE_DT')
    daily_return = LoadData.get_df('adj_pct_chg', 'TRADE_DT',Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)
    daily_close = LoadData.get_df('S_DQ_CLOSE' ,'TRADE_DT', Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)
    daily_shares_a = LoadData.get_df('S_SHARE_TOTALA' ,'TRADE_DT', Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)
    daily_shares_total = LoadData.get_df('TOT_SHR' ,'TRADE_DT',Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)

    # load income data
    income_statement = pd.read_pickle('./Data/income.pkl')
    idlist = ['S_INFO_WINDCODE', 'ANN_DT', 'REPORT_PERIOD', 'NET_PROFIT_INCL_MIN_INT_INC', 'PLUS_NON_OPER_REV', 'LESS_NON_OPER_EXP' ]
    income_statement = income_statement[idlist]
    income_statement = income_statement[~income_statement['ANN_DT'].isnull()]
    income_statement['ANN_DT'] = income_statement['ANN_DT'].astype(int)
    income_statement['ANN_DT'] = income_statement['ANN_DT'].astype(str)
    # fillna
    income_statement['PLUS_NON_OPER_REV'].fillna(0, inplace=True)
    income_statement['LESS_NON_OPER_EXP'].fillna(0, inplace=True)
    income_statement['Earnings'] = income_statement['NET_PROFIT_INCL_MIN_INT_INC'] + income_statement['PLUS_NON_OPER_REV'] - income_statement['LESS_NON_OPER_EXP']
    income_statement = income_statement.reset_index(drop=True)

    season_index = income_statement['REPORT_PERIOD'].unique()
    season_index.sort()
    stock_list_1 = income_statement['S_INFO_WINDCODE'].unique()

    income_locate_cum = LoadData.get_locate_cum(income_statement, 'REPORT_PERIOD')
    df_announce_date = LoadData.get_df('ANN_DT', 'REPORT_PERIOD' ,income_statement, season_index, stock_list_1, income_locate_cum, start_day, end_day)
    df_quarterly_earnings = LoadData.get_df('Earnings', 'REPORT_PERIOD' ,income_statement, season_index, stock_list_1, income_locate_cum, start_day, end_day)

else:
    daily_return = pd.read_pickle('./Data/daily_return.pkl')
    daily_close = pd.read_pickle('./Data/daily_close.pkl')
    daily_shares_a = pd.read_pickle('./Data/daily_shares_a.pkl')
    daily_shares_total = pd.read_pickle('./Data/daily_shares_total.pkl')

    df_announce_date = pd.read_pickle('./Data/df_announce_date.pkl')
    df_quarterly_earnings = pd.read_pickle('./Data/df_quarterly_earnings.pkl')

#%% Monthly

df_monthly_return = Daily2Monthly.cal_monthly_return(daily_return)
df_monthly_shares_a = Daily2Monthly.generate_monthly_df(daily_shares_a)
df_monthly_shares_total = Daily2Monthly.generate_monthly_df(daily_shares_total)


# %%  Size (A-share cap)

df_price = Daily2Monthly.generate_monthly_df(daily_close)
df_size = df_price*df_monthly_shares_a  # A-share capitalization

# %%  Value

# seasonal diff
df_quarterly_earnings_diff = df_quarterly_earnings.apply(lambda s: LoadIncome.season_diff(s))

# announce date adjustment
monthly_timeline = Daily2Monthly.generate_monthly_index(df_monthly_return.index)
earnings_path = './Data/df_monthly_earnings_adjusted_DIFF.pkl'
df_monthly_earnings = LoadIncome.adjusted_financials(df_announce_date, df_quarterly_earnings_diff, earnings_path, monthly_timeline )

df_monthly_earnings = df_monthly_earnings.loc[:,stock_list]
df_value = df_monthly_earnings/(df_price*df_monthly_shares_total)


# %% Condition Filters



