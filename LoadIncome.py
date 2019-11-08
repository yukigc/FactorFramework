#%% 
import os
import copy
import datetime
import numpy as np
import pandas as pd

def adjusted_financials(df_announce_date, df_quarterly_earnings, result_path, monthly_timeline ):
   
    if os.path.exists(result_path):
        df_monthly_earnings = pd.read_pickle(result_path)
    else:
        
        df_monthly_earnings = financials_announced(df_quarterly_earnings, df_announce_date, monthly_timeline)
        df_monthly_earnings.to_pickle(result_path)
        
    return df_monthly_earnings


def financials_announced(df_quarterly_financials, df_announce_date, monthly_timeline):
    '''
    get the latest announced data before month's end
    '''
    announced_result = pd.DataFrame(index = monthly_timeline, columns = df_quarterly_financials.columns)
    for stock in df_quarterly_financials.columns:
        print(stock)
        announced_date = df_announce_date.loc[:,stock]
        date_trans = announced_date[announced_date.notnull()].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
        announced_date[date_trans.index] = date_trans
        announced_date = announced_date.astype('datetime64[ns]')
        announced_result[stock] = [df_quarterly_financials.loc[(announced_date[announced_date<t].index[-1]),stock] if len(announced_date[announced_date<t].index) else np.nan for t in announced_result.index]
    return announced_result


def season_diff(s):
    '''
    season difference
    before 2002: half year report
    '''
    result = copy.deepcopy(s)
    result['1999-03-31'] = np.nan
    result['2000-03-31'] = np.nan
    result['2000-09-30'] = np.nan
    result['2001-03-31'] = np.nan
    result['2001-09-30'] = np.nan
    #result['1998-12-31'] = s['1998-12-31'] - s['1998-06-30']
    result['1999-12-31'] = s['1999-12-31'] - s['1999-06-30']
    result['2000-12-31'] = s['2000-12-31'] - s['2000-06-30']
    result['2001-12-31'] = s['2001-12-31'] - s['2001-06-30']

    for t in range(0, len(s.index)):
        if s.index[t].year >= 2002 and s.index[t].month != 3 :
            result[s.index[t]] = s[s.index[t]] - s[s.index[t-1]]
            
    return result

