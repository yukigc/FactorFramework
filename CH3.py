
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
import Filters
import FactorDecile
import ConstructPort

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
    daily_trade_volume = LoadData.get_df('S_DQ_VOLUME' ,'TRADE_DT',Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum, start_day, end_day)

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
    daily_trade_volume = pd.read_pickle('./Data/daily_trade_volume.pkl')

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

df_ipo_date_new = Filters.load_ipo_date('./Data/IPO_date.xlsx')
df_ipo_filter = Filters.IPO_6m_filters(df_ipo_date_new, df_monthly_return)
df_trade_filters = Filters.trading_records_filters(daily_trade_volume, df_monthly_return, 0.75, 0.5, if_plot_ts = False)
df_size_filters = Filters.size_filters(df_size, (df_ipo_filter&df_trade_filters), bottom_percentile = 0.3, if_plot_ratios = False)
df_value_filters = Filters.value_filters(df_value)
df_final_filters = df_size_filters & df_value_filters

# %%

df_size_decile = FactorDecile.size_decile(df_size, df_filters = df_final_filters)
df_value_decile = FactorDecile.value_decile(df_value, df_filters = df_final_filters)

factors_return = pd.DataFrame(index = df_monthly_return.index, columns = ['MKT', 'SMB', 'VMG'] )
factors_return['MKT'] = ConstructPort.value_weighted(df_monthly_return, df_size, df_filters = df_final_filters)

value_namelist = ['S/V','S/M','S/G','B/V','B/M','B/G']
crosssec_return_CH3, cross_cnt_CH3 = ConstructPort.crosssec_portfolios(df_monthly_return, df_size, df_size_decile, df_value_decile, value_namelist)

factors_return['SMB'] = (crosssec_return_CH3['S/V']+crosssec_return_CH3['S/M']+crosssec_return_CH3['S/G'] - \
                         crosssec_return_CH3['B/V']-crosssec_return_CH3['B/M']-crosssec_return_CH3['B/G'])/3
factors_return['VMG'] = (crosssec_return_CH3['S/V']+crosssec_return_CH3['B/V']-crosssec_return_CH3['S/G']-crosssec_return_CH3['B/G'])/2

#  CH4
daily_turnover = daily_trade_volume/daily_shares_total
df_turnover_decile = FactorDecile.turnover_decile(daily_turnover, df_filters = df_final_filters)

turnover_namelist = ['S/P','S/M','S/O','B/P','B/M','B/O']
crosssec_return_CH4, cross_cnt_CH4 = ConstructPort.crosssec_portfolios(df_monthly_return, df_size, df_size_decile, df_turnover_decile, turnover_namelist)

factors_return['SMB_1'] = (crosssec_return_CH4['S/P']+crosssec_return_CH4['S/M']+crosssec_return_CH4['S/O'] - \
                         crosssec_return_CH4['B/P']-crosssec_return_CH4['B/M']-crosssec_return_CH4['B/O'])/3
factors_return['PMO'] = (crosssec_return_CH4['S/P']+crosssec_return_CH4['B/P']-crosssec_return_CH4['S/O']-crosssec_return_CH4['B/O'])/2

factors_return['SMB_new'] = (factors_return['SMB']+factors_return['SMB_1'])/2

#%%
factors_return = factors_return.loc['2000-01-01':,:]

rf_series = ConstructPort.load_rf_series('./Data/一年期定期存款利率.csv', factors_return.index)
factors_return['MKT'] = factors_return['MKT'] - rf_series['rf']

factors_return = factors_return.astype(float)
corr_matx = factors_return[['MKT', 'SMB', 'VMG']].corr(method='pearson')

table3 = pd.DataFrame(index = ['MKT', 'SMB', 'VMG'], columns=['Mean', 'Std', 't-stat', 'MKT', 'SMB', 'VMG'])
table3['Mean'] = factors_return.mean()*100
table3['Std'] = factors_return.std()*100
table3['t-stat'] = np.sqrt(factors_return['MKT'].count())*table3['Mean']/table3['Std']
table3[['MKT', 'SMB', 'VMG']] = corr_matx.values

corr_matx_new = factors_return[['MKT', 'SMB_new', 'VMG', 'PMO']].corr(method='pearson')
table_CH4 = pd.DataFrame(index = ['MKT', 'SMB_new', 'VMG', 'PMO'], columns=['Mean', 'Std', 't-stat', 'MKT', 'SMB_new', 'VMG', 'PMO'])
table_CH4['Mean'] = factors_return.mean()*100
table_CH4['Std'] = factors_return.std()*100
table_CH4['t-stat'] = np.sqrt(factors_return['MKT'].count())*table_CH4['Mean']/table_CH4['Std']
table_CH4[['MKT', 'SMB_new', 'VMG', 'PMO']] = corr_matx_new.values


#%%  corr check

CH3_path = './Data/CH3_LSY_.csv'
CH3 = pd.read_csv(CH3_path)
CH3.index = CH3.iloc[:,0].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))
CH3 = CH3[['mktrf', 'SMB', 'VMG']]

print('Corrcoef')
print('MKT: ', np.corrcoef(CH3['mktrf'][1:], factors_return['MKT'][1:]*100)[0][1])
print('SMB: ', np.corrcoef(CH3['SMB'][1:], factors_return['SMB'][1:]*100)[0][1])
print('VMG: ', np.corrcoef(CH3['VMG'][1:], factors_return['VMG'][1:]*100)[0][1])

