"""Preprocess acquired data.

This python file includes procedure that first cleans the data from missing values and outliers then provides useful
visualisation of the data and finally transforms the data using normalization to improve the forecasting performance.

Typical usage example:

    preprocessed_data = preprocess('stocks')

"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
from scipy import stats
from src.utils import create_dir
import datetime


def preprocess(category):
    """Preprocess data obtained from storage.

    Treat each dataset separately. Since each dataset has its own distribution features, different preprocessing
    methods are applied to each dataset respectively.


    Args:
        category: The name of specific dataset.

    """
    print("preprocess!!!")

    create_dir('visualization')
    if category == 'weather':
        return preprocess_weather_data()
    elif category == 'stocks':
        return preprocess_stocks_data()
    elif category == 'covid':
        return preprocess_covid_data()


def preprocess_covid_data():
    """Preprocess covid-19 data.

    First conduct data cleaning by imputing to deal with missing values. Then provide data visualization. Finally,
    transform the data by normalization.

    Returns:
        Preprocessed dataframe for covid-19 data.

    """
    covid_df = pd.read_json('data/covid_data.json')
    covid_df = covid_df.sort_values(by=['submission_date'])
    covid_df = covid_df.drop(
        columns=['state', 'created_at', 'consent_cases', 'consent_deaths', 'pnew_case', 'pnew_death'])
    # set index
    covid_df = covid_df.set_index('submission_date')
    extra_df = covid_df.iloc[:1].copy()
    # cases should be zero before there is a statistics about covid
    for column in extra_df.columns:
        extra_df.loc[:, column] = 0
    cur_date_string = '2017-04-01'
    cur_date = datetime.datetime.strptime('2017-04-01', '%Y-%m-%d').date()
    covid_start_date = covid_df.iloc[:1].index[0]
    extra_df = extra_df.rename(index={covid_start_date: cur_date_string}, inplace=False)
    covid_start_date = datetime.datetime.strptime(covid_start_date, '%Y-%m-%d').date()
    while cur_date < covid_start_date:
        covid_df = pd.concat([extra_df, covid_df.loc[:]])
        extra_df = extra_df.rename(
            index={cur_date.strftime('%Y-%m-%d'): (cur_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')},
            inplace=False)
        cur_date = cur_date + datetime.timedelta(days=1)
    covid_df = covid_df.sort_index()
    covid_df = covid_df.set_index(covid_df.index.astype(dtype='datetime64'))
    # 2. provide visualization of the data
    # visualization through box plot, clearly no outlier is detected
    for column in covid_df:
        show_boxplot(covid_df, column, 'covid')

    # visualization through z-score, clearly no outlier is detected(z-score > 3)
    for column in covid_df:
        plot_zscore(covid_df, column, 'covid')
    # 3. transformation
    covid_df = normalize(covid_df)
    return covid_df


def show_weather(weather_df, column, description):
    """Plot line chart for weather data and save in local disk.

    Args:
        weather_df: Pandas dataframe for weather data.
        column: Specific feature chosen from dataframe.
        description: Extra words to describe the data.

    """
    plt.figure()
    plt.plot(weather_df[column], label=column,
             linestyle='-', c='y')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.xlim([datetime.date(2017, 4, 1), datetime.date(2022, 4, 1)])
    plt.savefig('visualization/weather_' + column.lower() + description + '.png')
    plt.close()


def show_stocks(stocks_df, column):
    """Plot line chart for stocks data and save in local disk.

    Args:
        stocks_df: Pandas dataframe for stocks data.
        column: Specific feature chosen from dataframe.

    """
    plt.figure()
    label = column + ' value'
    plt.plot(stocks_df[column], label=label,
             linestyle='-', c='r')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.savefig('visualization/stock_' + column.lower() + '.png')
    plt.close()


def show_close_season(stocks_df):
    """Plot to show seasonality for data and save the figure in local disk.

    Args:
        stocks_df: Pandas dataframe.

    """
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
    plt.savefig('visualization/stock_close_separate.png')
    plt.close()


def show_boxplot(df, column, category):
    """Provide boxplot and save to the local disk.

    Args:
        df: Pandas dataframe.
        column: Specific chosen feature of the data.
        category: Name describing the dataset.

    """
    plt.figure()
    data_array = df[column]
    sns.boxplot(data_array)
    y_label = column + ' Value'
    plt.ylabel(y_label)
    plt.savefig('visualization/' + category + '_' + column.lower() + '_box.png')
    plt.close()


def plot_zscore(df, column, category):
    """Provide Z-score plot and save to the local disk.

    Args:
        df: Pandas dataframe.
        column: Specific chosen feature of the data.
        category: Name describing the dataset.

    """
    plt.figure()
    z = np.abs(stats.zscore(df[column]))
    plt.plot(z)
    plt.grid()
    plt.ylabel('Z score')
    plt.savefig('visualization/' + category + '_' + column.lower() + '_zscore.png')
    plt.close()


def preprocess_stocks_data():
    """Preprocess stocks data.

    First conduct data cleaning by interpolating to deal with missing values. Then provide data visualization.
    Finally, transform the data by normalization.

    Returns:
        Preprocessed dataframe for stocks data

    """
    stocks_df = pd.read_json('data/stocks_data.json')
    stocks_df = stocks_df.drop('Stock Splits', axis=1)
    stocks_df = stocks_df.drop('Dividends', axis=1)

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

    last_line_df = stocks_df.iloc[-1:]
    last_line_idx = last_line_df.index[-1]

    # change index 2017-04-03 to 2017-04-01
    missing_day_one = first_line_df
    new_index = first_line_idx - datetime.timedelta(days=2)
    missing_day_one = missing_day_one.rename(index={first_line_idx: new_index}, inplace=False)

    # change index 2017-04-03 to 2017-04-02
    missing_day_two = first_line_df
    new_index = first_line_idx - datetime.timedelta(days=1)
    missing_day_two = missing_day_two.rename(index={first_line_idx: new_index}, inplace=False)

    missing_day_last = last_line_df
    new_index = last_line_idx + datetime.timedelta(days=1)
    missing_day_last = missing_day_last.rename(index={last_line_idx: new_index}, inplace=False)

    stocks_df = pd.concat([missing_day_one, missing_day_two, stocks_df.loc[:], missing_day_last])

    print(stocks_df)
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
    """Data normalization

    Args:
        df: Pandas dataframe

    Returns:
        Normalized dataframe

    """
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value + 1)
    return result


def perform_binning(weather_df):
    """Perform data binning to dataframe

    Args:
        weather_df: Pandas dataframe for weather data

    Returns:
        Smoothed data in Pandas dataframe format

    """
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
    """Preprocess weather data.

    First conduct data cleaning by imputing to deal with missing values. Then provide binning to smooth the data and
    provide data visualization. Finally, transform the data by normalization.

    Returns:
        Preprocessed dataframe for weather data

    """
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
    fig_copy.savefig('visualization/weather_missing_plot.png', bbox_inches='tight')
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
        show_weather(weather_df, column, '_imputed')

    # Perform data binning
    weather_df = perform_binning(weather_df)
    # drop bins

    # 3. visualization
    # plot to see original data distribution
    for column in original_weather_df:
        show_weather(original_weather_df, column, '_ori')
    for column in weather_df:
        show_weather(weather_df, column, '_binned')
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
