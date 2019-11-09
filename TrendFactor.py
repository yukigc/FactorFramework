#%%import sys
import numpy as np
import pandas as pd
import datetime
import math
import copy
from tqdm import tqdm
import LoadData
import Daily2Monthly
import LoadIncome
import Filters
import FactorDecile
import ConstructPort
import statsmodels.api as sm

tqdm.pandas(desc='pandas bar')

# if is not initial, load pkl files directly
is_initial = 0

# %%

# back test period
start_day = '2000-01-01'
end_day = '2018-12-31'

# load price,share data
# 复权收盘价 = 收盘价*复权因子
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
    daily_trade_volume = LoadData.get_df('S_DQ_VOLUME' ,'TRADE_DT',Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)
    daily_adjfactor = LoadData.get_df('S_DQ_ADJFACTOR' ,'TRADE_DT',Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)

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

    # save pkl
    daily_return.to_pickle('./Data_trend/daily_return.pkl')
    daily_close.to_pickle('./Data_trend/daily_close.pkl')
    daily_shares_a.to_pickle('./Data_trend/daily_shares_a.pkl')
    daily_shares_total.to_pickle('./Data_trend/daily_shares_total.pkl')
    daily_trade_volume.to_pickle('./Data_trend/daily_trade_volume.pkl')
    daily_adjfactor.to_pickle('./Data_trend/daily_adjfactor.pkl')

    df_announce_date.to_pickle('./Data_trend/df_announce_date.pkl')
    df_quarterly_earnings.to_pickle('./Data_trend/df_quarterly_earnings.pkl')


else:
    daily_return = pd.read_pickle('./Data_trend/daily_return.pkl')
    daily_close = pd.read_pickle('./Data_trend/daily_close.pkl')
    daily_shares_a = pd.read_pickle('./Data_trend/daily_shares_a.pkl')
    daily_shares_total = pd.read_pickle('./Data_trend/daily_shares_total.pkl')
    daily_trade_volume = pd.read_pickle('./Data_trend/daily_trade_volume.pkl')
    daily_adjfactor = pd.read_pickle('./Data_trend/daily_adjfactor.pkl')

    df_announce_date = pd.read_pickle('./Data_trend/df_announce_date.pkl')
    df_quarterly_earnings = pd.read_pickle('./Data_trend/df_quarterly_earnings.pkl')


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
earnings_path = './Data_trend/df_monthly_earnings_adjusted_DIFF.pkl'
df_monthly_earnings = LoadIncome.adjusted_financials(df_announce_date, df_quarterly_earnings_diff, earnings_path, monthly_timeline )

df_monthly_earnings = df_monthly_earnings.loc[:,stock_list]
df_value = df_monthly_earnings/(df_price*df_monthly_shares_total)

# %% Condition Filters

df_ipo_date_new = Filters.load_ipo_date('./Data/IPO_date.xlsx')
df_ipo_filter = Filters.IPO_6m_filters(df_ipo_date_new, df_monthly_return)
df_size_filters = Filters.size_filters(df_size, df_ipo_filter, bottom_percentile = 0.3, if_plot_ratios = False)
df_final_filters = df_size_filters


# %%

# MKT: 0.91
factors_return = pd.DataFrame(index = df_monthly_return.index, columns = ['MKT', 'SMB', 'VMG', 'Trend'] )
factors_return['MKT'] = ConstructPort.value_weighted(df_monthly_return, df_size, df_filters = df_final_filters)
rf_series = ConstructPort.load_rf_series('./Data/一年期定期存款利率.csv', factors_return.index)
factors_return['MKT'] = factors_return['MKT'] - rf_series['rf']

# %% Trend

# Lag: 3,5,10,20,50,100,200,300,400

# Moving Average signal: price
daily_adj_close = daily_close * daily_adjfactor
daily_adj_close = daily_adj_close.fillna(method='ffill')

MP3 = (daily_adj_close.rolling(window=3).mean()/daily_adj_close).loc[monthly_timeline,:]
MP5 = (daily_adj_close.rolling(window=5).mean()/daily_adj_close).loc[monthly_timeline,:]
MP10 = (daily_adj_close.rolling(window=10).mean()/daily_adj_close).loc[monthly_timeline,:]
MP20 = (daily_adj_close.rolling(window=20).mean()/daily_adj_close).loc[monthly_timeline,:]
MP50 = (daily_adj_close.rolling(window=50).mean()/daily_adj_close).loc[monthly_timeline,:]
MP100 = (daily_adj_close.rolling(window=100).mean()/daily_adj_close).loc[monthly_timeline,:]
MP200 = (daily_adj_close.rolling(window=200).mean()/daily_adj_close).loc[monthly_timeline,:]
MP300 = (daily_adj_close.rolling(window=300).mean()/daily_adj_close).loc[monthly_timeline,:]
MP400 = (daily_adj_close.rolling(window=400).mean()/daily_adj_close).loc[monthly_timeline,:]

# %%

# Moving Average signal: volume
# For a given lag, more than half of the days of trading records, and have traded this month

pd.options.mode.use_inf_as_na = True

def MV_lag(lag, df_monthly_return, daily_trade_volume):
    MV = pd.DataFrame(index = df_monthly_return.index, columns = df_monthly_return.columns)
    is_trade = daily_trade_volume.notnull() & (daily_trade_volume!=0)
    lag_records = is_trade.rolling(window=lag).sum() > (lag/2)
    # 每个月交易日数量，近似
    month_records = is_trade.rolling(window=22).sum() >= 1
    valid_records = (lag_records & month_records).loc[MV.index,:]
    MV = (daily_trade_volume.rolling(window=lag).sum().loc[MV.index,:]/lag)/daily_trade_volume.loc[MV.index,:]
    MV = MV[valid_records]
    MV = MV.fillna(method='ffill')
    return MV

MV3 = MV_lag(3, df_monthly_return, daily_trade_volume)
MV5 = MV_lag(5, df_monthly_return, daily_trade_volume)
MV10 = MV_lag(10, df_monthly_return, daily_trade_volume)
MV20 = MV_lag(20, df_monthly_return, daily_trade_volume)
MV50 = MV_lag(50, df_monthly_return, daily_trade_volume)
MV100 = MV_lag(100, df_monthly_return, daily_trade_volume)
MV200 = MV_lag(200, df_monthly_return, daily_trade_volume)
MV300 = MV_lag(300, df_monthly_return, daily_trade_volume)
MV400 = MV_lag(400, df_monthly_return, daily_trade_volume)


# %% 

# crossectional regression (2001-09-28)

df_coefficient = pd.DataFrame(index = df_monthly_return.index, 
                 columns=['MP3', 'MP5', 'MP10', 'MP20', 'MP50', 'MP100', 'MP200', 'MP300', 'MP400',
                          'MV3', 'MV5', 'MV10', 'MV20', 'MV50', 'MV100', 'MV200', 'MV300', 'MV400'])

regression_index = df_monthly_return.loc['2001-09-28':,:].index

for i in range(len(regression_index)-1):
    t = regression_index[i]
    tnext = regression_index[i+1]
    print(t)

    X = pd.concat([MP3.loc[t,:], MP5.loc[t,:], MP10.loc[t,:], 
                MP20.loc[t,:],MP50.loc[t,:], MP100.loc[t,:], 
                MP200.loc[t,:], MP300.loc[t,:], MP400.loc[t,:],
                MV3.loc[t,:], MV5.loc[t,:], MV10.loc[t,:], 
                MV20.loc[t,:],MV50.loc[t,:], MV100.loc[t,:], 
                MV200.loc[t,:], MV300.loc[t,:], MV400.loc[t,:]],axis=1)
    X.columns = ['MP3', 'MP5', 'MP10', 'MP20', 'MP50', 'MP100', 'MP200', 'MP300', 'MP400',
                'MV3', 'MV5', 'MV10', 'MV20', 'MV50', 'MV100', 'MV200', 'MV300', 'MV400']
    filter_stocks = df_final_filters.loc[t,:]
    filter_stocks = filter_stocks[filter_stocks].index
    X = X.loc[filter_stocks,:]
    X = X.dropna()
    # next month's return
    y = (df_monthly_return.loc[tnext,:])[X.index]

    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    df_coefficient.loc[tnext,:] = results.params[df_coefficient.columns]

#%%

# moving average coefficient
moving_lambda = 0.02
df_forecast_coefficient = pd.DataFrame(index = df_monthly_return.index, 
                 columns=['MP3', 'MP5', 'MP10', 'MP20', 'MP50', 'MP100', 'MP200', 'MP300', 'MP400',
                          'MV3', 'MV5', 'MV10', 'MV20', 'MV50', 'MV100', 'MV200', 'MV300', 'MV400'])

df_forecast_coefficient.loc['2001-10-31',:] = df_coefficient.loc['2001-10-31',:]
moving_index = df_monthly_return.loc['2001-10-31':,:].index
for i in range(1,len(moving_index)):
    df_forecast_coefficient.loc[moving_index[i],:] = (1-moving_lambda)*df_forecast_coefficient.loc[moving_index[i-1],:]+\
                                                    moving_lambda*df_coefficient.loc[moving_index[i],:]

#%%

# Expected Return

df_expected_return = pd.DataFrame(index = df_monthly_return.index, columns = df_monthly_return.columns)

expected_MP3 = MP3.multiply(df_forecast_coefficient.loc[:,'MP3'],axis=0)
expected_MP5 = MP5.multiply(df_forecast_coefficient.loc[:,'MP5'],axis=0)
expected_MP10 = MP10.multiply(df_forecast_coefficient.loc[:,'MP10'],axis=0)
expected_MP20 = MP20.multiply(df_forecast_coefficient.loc[:,'MP20'],axis=0)
expected_MP50 = MP50.multiply(df_forecast_coefficient.loc[:,'MP50'],axis=0)
expected_MP100 = MP100.multiply(df_forecast_coefficient.loc[:,'MP100'],axis=0)
expected_MP200 = MP200.multiply(df_forecast_coefficient.loc[:,'MP200'],axis=0)
expected_MP300 = MP300.multiply(df_forecast_coefficient.loc[:,'MP300'],axis=0)
expected_MP400 = MP400.multiply(df_forecast_coefficient.loc[:,'MP400'],axis=0)

expected_MV3 = MV3.multiply(df_forecast_coefficient.loc[:,'MV3'],axis=0)
expected_MV5 = MV5.multiply(df_forecast_coefficient.loc[:,'MV5'],axis=0)
expected_MV10 = MV10.multiply(df_forecast_coefficient.loc[:,'MV10'],axis=0)
expected_MV20 = MV20.multiply(df_forecast_coefficient.loc[:,'MV20'],axis=0)
expected_MV50 = MV50.multiply(df_forecast_coefficient.loc[:,'MV50'],axis=0)
expected_MV100 = MV100.multiply(df_forecast_coefficient.loc[:,'MV100'],axis=0)
expected_MV200 = MV200.multiply(df_forecast_coefficient.loc[:,'MV200'],axis=0)
expected_MV300 = MV300.multiply(df_forecast_coefficient.loc[:,'MV300'],axis=0)
expected_MV400 = MV400.multiply(df_forecast_coefficient.loc[:,'MV400'],axis=0)

df_expected_return = expected_MP3 + expected_MP5 + expected_MP10 +\
                     expected_MP20 + expected_MP50 + expected_MP100 +\
                     expected_MP200 + expected_MP300 + expected_MP400 +\
                     expected_MV3 + expected_MV5 + expected_MV10 +\
                     expected_MV20 + expected_MV50 + expected_MV100 +\
                     expected_MV200 + expected_MV300 + expected_MV400 

#%%

df_size_decile = FactorDecile.size_decile(df_size, df_filters = df_final_filters)
df_value_decile = FactorDecile.value_decile(df_value, df_filters = df_final_filters)
df_trend_decile = FactorDecile.trend_decile(df_expected_return, df_filters = df_final_filters)


# 4-factor model, triple sort
portfolio_namelist = ['S/V/H','S/V/M','S/V/L',
                      'S/M/H','S/M/M','S/M/L',
                      'S/G/H','S/G/M','S/G/L',
                      'B/V/H','B/V/M','B/V/L',
                      'B/M/H','B/M/M','B/M/L',
                      'B/G/H','B/G/M','B/G/L']

def triple_crosssec_portfolios(df_monthly_return, df_cap, df_size_decile, df_value_decile, df_trend_decile, namelist):
    '''
    triple sort, 2*3*3
    '''
    cross_return = pd.DataFrame(index = df_monthly_return.index, columns = namelist )
    cross_cnt = pd.DataFrame(index = df_monthly_return.index, columns = namelist )
    for t in range(len(cross_return.index)-1):
        for c in cross_return.columns:
            set_1 = set(df_size_decile.iloc[t,:][df_size_decile.iloc[t,:]==(c.split('/')[0])].index)
            set_2 = set(df_value_decile.iloc[t,:][df_value_decile.iloc[t,:]==(c.split('/')[1])].index)
            set_3 = set(df_trend_decile.iloc[t,:][df_trend_decile.iloc[t,:]==(c.split('/')[2])].index)
            p_list = list(set_3.intersection(set_1.intersection(set_2)))
            p_weight = df_cap.loc[cross_return.index[t], p_list]
            p_next_return = df_monthly_return.loc[cross_return.index[t+1], p_list]
            cross_return.loc[cross_return.index[t+1], c] = (p_weight*p_next_return).sum()/p_weight.sum()
            #print((p_weight*p_next_return).sum()/p_weight.sum())
            cross_cnt.loc[cross_return.index[t+1], c] = len(p_list)
    return cross_return, cross_cnt

crosssec_return_Trend, cross_cnt_Trend = triple_crosssec_portfolios(df_monthly_return, df_size, df_size_decile, df_value_decile, df_trend_decile, portfolio_namelist)

#%%

# factor construction

factors_return['SMB'] = crosssec_return_Trend[['S/V/H','S/V/M','S/V/L','S/M/H','S/M/M','S/M/L','S/G/H','S/G/M','S/G/L']].mean(axis=1) -\
                        crosssec_return_Trend[['B/V/H','B/V/M','B/V/L','B/M/H','B/M/M','B/M/L','B/G/H','B/G/M','B/G/L']].mean(axis=1)

factors_return['VMG'] = crosssec_return_Trend[['S/V/H','S/V/M','S/V/L','B/V/H','B/V/M','B/V/L']].mean(axis=1) -\
                        crosssec_return_Trend[['S/G/H','S/G/M','S/G/L','B/G/H','B/G/M','B/G/L']].mean(axis=1)

factors_return['Trend'] = crosssec_return_Trend[['S/V/H','S/M/H','S/G/H','B/V/H','B/M/H','B/G/H']].mean(axis=1) -\
                          crosssec_return_Trend[['S/V/L','S/M/L','S/G/L','B/V/L','B/M/L','B/G/L']].mean(axis=1)

factors_return = factors_return.loc['2005-01-01':'2018-07-31',['Trend','MKT','SMB','VMG']]


#%%  Summary

def maxdrawdown(arr):
	i = np.argmax((np.maximum.accumulate(arr) - arr)/np.maximum.accumulate(arr)) # end of the period
	j = np.argmax(arr[:i]) # start of period
	return (1-arr[i]/arr[j])

summary_A = pd.DataFrame(index=['Mean','Std dev','Shapre ratio','Skewness','MDD'], 
                        columns = ['Trend','MKT','SMB','VMG','PMO'])

summary_A.loc['Mean',['Trend','MKT','SMB','VMG']] = factors_return.mean()*100
summary_A.loc['Std dev',['Trend','MKT','SMB','VMG']] = factors_return.std()*100
summary_A.loc['Shapre ratio',:] = summary_A.loc['Mean',:]/summary_A.loc['Std dev',:] 
summary_A.loc['Skewness',['Trend','MKT','SMB','VMG']] = factors_return.skew()
summary_A.loc['MDD',['Trend','MKT','SMB','VMG']] = factors_return[['Trend','MKT','SMB','VMG']].apply(lambda s: maxdrawdown(np.array((1+s).cumprod())))*100


#%%
'''
lag = 400
MV400 = pd.DataFrame(index = df_monthly_return.index, columns = df_monthly_return.columns)

def MA_volume_s(s, t, lag, daily_trade_volume):
    loc_s = daily_trade_volume.columns.get_loc(s.name)
    loc_t = daily_trade_volume.index.get_loc(t.name)

    trading_lag = daily_trade_volume.iloc[max(0,loc_t-400):loc_t,loc_s]
    is_trade_count = trading_lag.notnull().sum() - len(trading_lag[trading_lag==0])
    trading_month = daily_trade_volume.iloc[max(0,loc_t-30):loc_t,loc_s]
    is_this_month = trading_month.notnull().sum() - len(trading_month[trading_month==0])
    
    if is_trade_count>=(lag//2) and is_this_month>=1 :
        if daily_trade_volume.iloc[loc_t,loc_s]!=np.nan and daily_trade_volume.iloc[loc_t,loc_s]!=0:
            # mean over lag L?
            return (trading_lag.sum()/lag)/daily_trade_volume.iloc[loc_t,loc_s]
    return np.nan

tests = MV400.iloc[:,:100].progress_apply(lambda s: pd.DataFrame(s).apply(lambda t: MA_volume_s(s, t, 400, daily_trade_volume), axis=1), axis=0)

'''