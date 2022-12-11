import datetime

import pandas as pd
import matplotlib.pyplot as plt

from src import store


def preprocess(category):
    if category == 'weather':
        return preprocess_weather_data()
    elif category == 'stocks':
        return preprocess_stocks_data()

    print("preprocess!!!")


def preprocess_weather_data():
    # read as pandas dataframe
    weather_df = pd.read_json('data/weather_data.json')

    # data cleaning


def show_stocks(stocks_df):
    plt.figure()
    plt.plot(stocks_df['Open'], label='Open value',
             linestyle='-', c='r')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.savefig('demo/stock_open.png')

    plt.figure()
    plt.plot(stocks_df['Close'], label='Close value',
             linestyle='-', c='r')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.savefig('demo/stock_close.png')

    plt.figure()
    plt.plot(stocks_df['High'], label='High value',
             linestyle='-', c='r')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.savefig('demo/stock_high.png')

    plt.figure()
    plt.plot(stocks_df['Low'], label='Low value',
             linestyle='-', c='r')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.savefig('demo/stock_low.png')


def show_close_season(stocks_df):
    figure = plt.figure()
    plt.subplot(511)
    plt.plot(stocks_df['Close'].loc['2017-04-01 00:00:00':'2018-04-01 00:00:00'],
             linestyle='-', c='b')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.subplot(512)
    plt.plot(stocks_df['Close'].loc['2018-04-01 00:00:00':'2019-04-01 00:00:00'],
             linestyle='-', c='b')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.subplot(513)
    plt.plot(stocks_df['Close'].loc['2019-04-01 00:00:00':'2020-04-01 00:00:00'],
             linestyle='-', c='b')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.subplot(514)
    plt.plot(stocks_df['Close'].loc['2020-04-01 00:00:00':'2021-04-01 00:00:00'],
             linestyle='-', c='b')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.subplot(515)
    plt.plot(stocks_df['Close'].loc['2021-04-01 00:00:00':'2022-04-01 00:00:00'],
             linestyle='-', c='b')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.suptitle('Close value')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig('demo/stock_close_separate.png')


def preprocess_stocks_data():
    stocks_df = pd.read_json('data/stocks_data.json')

    # data is missing for weekends. set date as index in order to interpolate, and more importantly make data tidy
    stocks_df = stocks_df.set_index('Date')

    # show open, close, high, low values from 2017-04-01 to 2022-04-01
    show_stocks(stocks_df)

    # show close value for 5 years separately to find out if the trend is with seasonality
    show_close_season(stocks_df)

    # For handling missing data, it is a time-series problem
    # no clear seasonality for close value detected, thereby implement linear interpolation for missing values

    # upsample to one day interval
    stocks_df = stocks_df.resample('1D')

    # interpolate values (linear interpolate)
    stocks_df = stocks_df.interpolate(method='time')

    # Handling missing data at the start and end date
    first_line_df = stocks_df.iloc[:1]
    first_line_idx = first_line_df.index[0]

    # change index 2017-04-03 to 2017-04-01
    missing_day_one = first_line_df
    new_index = first_line_idx - datetime.timedelta(days=2)
    missing_day_one = missing_day_one.rename(index={first_line_idx: new_index}, inplace=False)

    # change index 2017-04-03 to 2017-04-02
    missing_day_two = first_line_df
    new_index = first_line_idx - datetime.timedelta(days=1)
    missing_day_two = missing_day_two.rename(index={first_line_idx: new_index}, inplace=False)
    stocks_df = pd.concat([missing_day_one, missing_day_two, stocks_df.loc[:]])

    print("...")


if __name__ == "__main__":
    preprocess_stocks_data()
