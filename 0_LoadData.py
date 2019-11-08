
#%%
import numpy as np
import pandas as pd
import datetime
import math
import copy


# %%
def get_df(variable, tname ,Daily_return_with_cap, timeindex_day, stock_list, stocks_locate_cum):
    df = pd.DataFrame(index = timeindex_day, columns = stock_list)
    df = df.loc['1999-01-01':'2016-12-31', :]

    def stock_match(variable,tname, Daily_return_with_cap, s, stocks_locate_cum, timeindex):
        temp = Daily_return_with_cap.loc[stocks_locate_cum.loc[s.name,'start']:stocks_locate_cum.loc[s.name,'end'], [tname,variable]]
        temp = temp.drop_duplicates(tname)
        temp_df = pd.DataFrame(data = temp[variable].values, index = temp[tname], columns = [variable])
        result = pd.DataFrame(index = timeindex, columns = [variable])
        result[variable] = temp_df[variable] 
        return result[variable].values

    df = df.apply(lambda s: stock_match(variable,tname,Daily_return_with_cap, s, stocks_locate_cum, df.index))
    return df
    

# %%
