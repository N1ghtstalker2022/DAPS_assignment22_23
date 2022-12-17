import copy
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from scipy import stats

from src import store


def preprocess(category):
    if category == 'weather':
        return preprocess_weather_data()
    elif category == 'stocks':
        return preprocess_stocks_data()

    print("preprocess!!!")


def show_weather(weather_df, column):
    plt.figure()
    plt.plot(weather_df[column], label=column,
             linestyle='-', c='y')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.xlim([datetime.date(2017, 4, 1), datetime.date(2022, 4, 1)])
    plt.savefig('demo/weather_' + column.lower() + '.png')
    plt.close()


def show_weather_ori(weather_df, column):
    plt.figure()
    plt.plot(weather_df[column], label=column,
             linestyle='-', c='y')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.xlim([datetime.date(2017, 4, 1), datetime.date(2022, 4, 1)])
    plt.savefig('demo/weather_' + column.lower() + '_ori.png')
    plt.close()


def show_weather_imputed(weather_df, column):
    plt.figure()
    plt.plot(weather_df[column], label=column,
             linestyle='-', c='y')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.xlim([datetime.date(2017, 4, 1), datetime.date(2022, 4, 1)])
    plt.savefig('demo/weather_' + column.lower() + '_imputed.png')
    plt.close()


def show_stocks(stocks_df, column):
    plt.figure()
    label = column + ' value'
    plt.plot(stocks_df[column], label=label,
             linestyle='-', c='r')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.savefig('demo/stock_' + column.lower() + '.png')
    plt.close()


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
    plt.close()


def show_boxplot(stocks_df, column, category):
    plt.figure()
    data_array = stocks_df[column]
    sns.boxplot(data_array)
    y_label = column + ' Value'
    plt.ylabel(y_label)
    plt.savefig('demo/' + category + '_' + column.lower() + '_box.png')
    plt.close()


def plot_zscore(stocks_df, column, category):
    plt.figure()
    z = np.abs(stats.zscore(stocks_df[column]))
    plt.plot(z)
    plt.grid()
    plt.ylabel('Z score')
    plt.savefig('demo/' + category + '_' + column.lower() + '_zscore.png')
    plt.close()


def preprocess_stocks_data():
    stocks_df = pd.read_json('data/stocks_data.json')
    stocks_df = stocks_df.drop('Stock Splits', axis=1)
    # data is missing for weekends. set date as index in order to interpolate, and more importantly make data tidy
    stocks_df = stocks_df.set_index('Date')

    # show open, close, high, low values from 2017-04-01 to 2022-04-01
    for column in stocks_df[['Open', 'Close', 'High', 'Low']]:
        show_stocks(stocks_df, column)

    # show close value for 5 years separately to find out if the trend is with seasonality
    show_close_season(stocks_df)

    # 1. For handling missing data, it is a time-series problem
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
    # 2. provide visualization of the data
    # visualization through box plot, clearly no outlier is detected
    for column in stocks_df[['Open', 'Close', 'High', 'Low']]:
        show_boxplot(stocks_df, column, 'stock')

    # visualization through z-score, clearly no outlier is detected(z-score > 3)
    for column in stocks_df[['Open', 'Close', 'High', 'Low']]:
        plot_zscore(stocks_df, column, 'stock')
    # 3. transformation
    stocks_df = normalize(stocks_df)
    return stocks_df

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value + 1)
    return result
def perform_binning(weather_df):
    bin_size = 3
    bin_map = {}
    for column in weather_df:
        smoothed_value_list = []
        single_bin_values = 0
        for i in range(len(weather_df)):
            cur_index = weather_df.index[i]
            single_bin_values += weather_df.loc[cur_index, column]
            if (i + 1) % bin_size == 0 or i + 1 == len(weather_df):
                smoothed_value = single_bin_values / bin_size
                smoothed_value_list.append(smoothed_value)
                single_bin_values = 0

        bin_map[column] = smoothed_value_list

    for column in weather_df:
        cur_smoothed_list = bin_map[column]
        j = 0
        for i in range(len(weather_df)):
            cur_index = weather_df.index[i]
            weather_df.loc[cur_index, column] = cur_smoothed_list[j]
            if (i + 1) % bin_size == 0:
                j += 1
    return weather_df


def preprocess_weather_data():
    # read as pandas dataframe
    weather_df = pd.read_json('data/weather_data.json')
    # weather_df = pd.read_json('../data/weather_data.json')

    # discard useless columns
    weather_df = weather_df.drop('STATION', axis=1)

    weather_df = weather_df.set_index('DATE')

    # data cleaning

    # 1. see the missing value distribution
    fig = msno.matrix(weather_df)
    fig_copy = fig.get_figure()
    fig_copy.savefig('plot.png', bbox_inches='tight')
    original_weather_df = copy.deepcopy(weather_df)
    # 2. handling missing values and noise
    # Imputing with median number
    for i in range(len(weather_df.index)):
        cur_index = weather_df.index[i]
        for column in weather_df.columns[:]:
            cur_value = weather_df.loc[cur_index, column]
            # implement cache thoughts
            if np.isnan(cur_value):
                cur_year = cur_index.year
                cur_month = cur_index.month
                cur_day = cur_index.day

                # get the value list for the same day of years from 2017 to 2022
                value_list = []
                if cur_year == 2017 or cur_month >= 4:
                    date_list = [pd.Timestamp(2017, cur_month, cur_day), pd.Timestamp(2018, cur_month, cur_day),
                                 pd.Timestamp(2019, cur_month, cur_day), pd.Timestamp(2020, cur_month, cur_day),
                                 pd.Timestamp(2021, cur_month, cur_day)]
                else:
                    date_list = [pd.Timestamp(2018, cur_month, cur_day), pd.Timestamp(2019, cur_month, cur_day),
                                 pd.Timestamp(2020, cur_month, cur_day), pd.Timestamp(2021, cur_month, cur_day),
                                 pd.Timestamp(2022, cur_month, cur_day)]

                for j in range(len(date_list)):
                    value = weather_df.loc[date_list[j], column]
                    if not np.isnan(value):
                        value_list.append(value)
                value_list.sort()

                # get median value
                list_len = len(value_list)
                if list_len % 2 != 0:
                    value_median = value_list[list_len // 2]
                else:
                    value_median = (value_list[list_len // 2 - 1] + value_list[list_len // 2]) / 2
                # complete replacing
                weather_df.loc[cur_index, column] = value_median

    # show imputed data
    for column in weather_df:
        show_weather_imputed(weather_df, column)

    # Perform data binning
    weather_df = perform_binning(weather_df)
    # drop bins

    # 3. visualization
    # plot to see original data distribution
    for column in original_weather_df:
        show_weather_ori(original_weather_df, column)
    for column in weather_df:
        show_weather(weather_df, column)
    #  through box plot

    for column in weather_df:
        show_boxplot(weather_df, column, 'weather')

    # visualization through z-score
    for column in weather_df:
        plot_zscore(weather_df, column, 'weather')

    # It seems a lot of outliers, but very possibly it is in line with reality, therefore we do not delete it for now
    # 4. Transformation
    # normalization
    weather_df = normalize(weather_df)
    return weather_df


if __name__ == "__main__":
    preprocess_weather_data()
