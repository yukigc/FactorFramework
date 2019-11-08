#%% 
import os
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
    announced_result = pd.DataFrame(index = monthly_timeline, columns = df_quarterly_financials.columns)
    for stock in df_quarterly_financials.columns:
        print(stock)
        announced_date = df_announce_date.loc[:,stock]
        date_trans = announced_date[announced_date.notnull()].apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d'))
        announced_date[date_trans.index] = date_trans
        announced_date = announced_date.astype('datetime64[ns]')
        announced_result[stock] = [df_quarterly_financials.loc[(announced_date[announced_date<t].index[-1]),stock] if len(announced_date[announced_date<t].index) else np.nan for t in announced_result.index]
    return announced_result