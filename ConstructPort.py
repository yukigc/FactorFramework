#%%
import os
import copy
import datetime
import numpy as np
import pandas as pd


def value_weighted(df_return, df_cap, df_filters):
    #df_mul = df_return*df_cap
    df_cap_prev = df_cap[df_filters].shift(1,axis=0)
    df_mul = df_return*df_cap_prev
    return df_mul.sum(axis=1)/df_cap_prev.sum(axis=1)
    
    
def load_rf_series(filepath, monthly_timeline):
    deposit_rate = pd.read_csv(filepath)
    # index datetime type
    deposit_rate.index = deposit_rate.iloc[:,0].apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
    result = pd.DataFrame(index = monthly_timeline, columns=['rf'])
    result['rf'] = [deposit_rate.iloc[:,1][deposit_rate.index<=t][-1] for t in result.index]
    #result['rf'] = (1+result['rf']/100).pow(1.0/12)-1
    result['rf'] = result['rf']/100
    return result


def crosssec_portfolios(df_monthly_return, df_cap, df_f1_decile, df_f2_decile, namelist):
    '''
    double sort
    '''
    cross_return = pd.DataFrame(index = df_monthly_return.index, columns = namelist )
    cross_cnt = pd.DataFrame(index = df_monthly_return.index, columns = namelist )
    for t in range(len(cross_return.index)-1):
        for c in cross_return.columns:
            set_1 = set(df_f1_decile.iloc[t,:][df_f1_decile.iloc[t,:]==c[0]].index)
            set_2 = set(df_f2_decile.iloc[t,:][df_f2_decile.iloc[t,:]==c[-1]].index)
            p_list = list(set_1.intersection(set_2))
            p_weight = df_cap.loc[cross_return.index[t], p_list]
            p_next_return = df_monthly_return.loc[cross_return.index[t+1], p_list]
            cross_return.loc[cross_return.index[t+1], c] = (p_weight*p_next_return).sum()/p_weight.sum()
            #print((p_weight*p_next_return).sum()/p_weight.sum())
            cross_cnt.loc[cross_return.index[t+1], c] = len(p_list)
    return cross_return, cross_cnt