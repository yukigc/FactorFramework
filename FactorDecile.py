#%%
import os
import copy
import datetime
import numpy as np
import pandas as pd


def size_decile(df_size, df_filters):
    df_size_decile = pd.DataFrame(index = df_size.index, columns = df_size.columns )
    def size_decile_rule(x, df_size, df_filters):
        # Small, Big
        test = df_size.loc[x.name,:]
        test = test[df_filters.loc[x.name,:]]
        x[list(test[test<=test.quantile(0.5)].index)] = 'S'
        x[list(test[test>test.quantile(0.5)].index)] = 'B'
        return x
    df_size_decile = df_size_decile.apply(lambda x: size_decile_rule(x, df_size, df_filters), axis=1)
    return df_size_decile



def value_decile(df_value, df_filters):
    df_value_decile = pd.DataFrame(index = df_value.index, columns = df_value.columns )
    def value_decile_rule(x, df_value, df_filters):
        # Growth, Middle, Value
        test = df_value.loc[x.name,:]
        test = test[df_filters.loc[x.name,:]]
        x[list(test[test<=test.quantile(0.3)].index)] = 'G'
        x[list(test[test>test.quantile(0.3)].index)] = 'M'
        x[list(test[test>test.quantile(0.7)].index)] = 'V'
        return x
    df_value_decile = df_value_decile.apply(lambda x: value_decile_rule(x, df_value, df_filters), axis=1)
    return df_value_decile


def turnover_decile(daily_turnover, df_filters):
    df_turnover_decile = pd.DataFrame(index = df_filters.index, columns = df_filters.columns )
    def turnover_decile_rule(x, df_turnover, df_filters):
        # Pessimistic, Middle, Optimistic
        test = df_turnover.loc[x.name,:]
        test = test[df_filters.loc[x.name,:]]
        x[list(test[test<=test.quantile(0.3)].index)] = 'P'
        x[list(test[test>test.quantile(0.3)].index)] = 'M'
        x[list(test[test>test.quantile(0.7)].index)] = 'O'
        return x
        
    turnover_avg20d = daily_turnover.rolling(window=20).mean()
    turnover_avg250d = daily_turnover.rolling(window=250).mean()
    turnover_abnormal_turnover = turnover_avg20d/turnover_avg250d
    df_abnormal_turnover = turnover_abnormal_turnover.loc[df_filters.index,:]
    
    df_turnover_decile = df_turnover_decile.apply(lambda x: turnover_decile_rule(x, df_abnormal_turnover, df_filters), axis=1)
    
    return df_turnover_decile