import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import datetime
import time


def get_change(stocks,start,end,interval):

    dicts = []
    for stock in stocks:
        
        time.sleep(.25) # yf starts returning nothing if it is hit too fast

        result = yf.download(stock, start= start, end= end ,interval = interval,prepost = True,threads = False)
        result['Change'] = (result['Close'] - result['Open'])/result['Open']
        result = result.dropna(axis=0, how='any') # when doing any interval greater than a day there are NAN values returned for days with dividend payout

        res_dict = dict(zip(result.index,result['Change']))
        res_dict['Stock'] = stock
        dicts.append(res_dict)
    
    df = pd.DataFrame(dicts)
    return df

def get_average(stocks,start,end,interval):

    dicts = []
    for stock in stocks:
        
        time.sleep(.25) # yf starts returning nothing if it is hit too fast

        result = yf.download(stock, start= start, end= end ,interval = interval,prepost = True,threads = False)
        result['Avg'] = (result['Close'] + result['Open'])/2
        result = result.dropna(axis=0, how='any') # when doing any interval greater than a day there are NAN values returned for days with dividend payout

        res_dict = dict(zip(result.index,result['Avg']))
        res_dict['Stock'] = stock
        dicts.append(res_dict)
    
    df = pd.DataFrame(dicts)
    return df

if __name__ == '__main__':

    # load companies
    comapines_path = 'Data\\Companies.csv'
    save_path = 'Data\\'
    companies = pd.read_csv(comapines_path)
    # companies = companies[companies.index < 50]

    # get daily change data
    start = datetime(2019,11,1)
    end = datetime(2020,5,1)
    daily_df = get_change(companies['Stock'],start,end,'1d')
    daily_df.to_pickle(save_path + 'daily.pkl')

    # get weekly change data
    start = datetime(2018,1,1)
    end = datetime(2020,5,1)
    weekly_df = get_change(companies['Stock'],start,end,'1wk')
    weekly_df.to_pickle(save_path + 'weekly.pkl')

    # get montly change data
    start = datetime(2016,1,1)
    end = datetime(2020,5,1)
    monthly_df = get_change(companies['Stock'],start,end,'1mo')
    monthly_df.to_pickle(save_path + 'monthly.pkl')

    # create test data
    start = datetime(2020,5,1)
    end = datetime(2020,7,21)
    test_daily_df = get_average(companies['Stock'],start,end,'1d')
    test_daily_df.to_pickle(save_path + 'test_daily.pkl')

    # create test data
    start = datetime(2020,5,1)
    end = datetime(2020,7,21)
    test_weekly_df = get_average(companies['Stock'],start,end,'1wk')
    test_weekly_df.to_pickle(save_path + 'test_weekly.pkl')
    

    