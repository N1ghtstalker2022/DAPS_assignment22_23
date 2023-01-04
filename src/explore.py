"""Exploratory data analysis.

This python file aims at providing exploratory data analysis to well preprocessed data. First, explore seasonality of
stocks data, Then, explore dependencies among year, month, week and day to find if there is some noise along with
fixed or predictable events. After this, explore the correlation between features of stocks data and the correlation
between stocks data and auxiliary data including weather and covid data. Finally, use hypothesis testing to better
understand the composition of your dataset and its representativeness.

Typical usage example:

    explore('data_dictionary')
"""
import random
from calendar import month_abbr, day_abbr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from src.utils import create_dir


def explore(data):
    """Explore data.

    Provide exploratory data analysis to preprocessed data. Explore main patterns and dependencies of the data.
    Provide hypothesis testing to gain some intuition about the data pattern.

    Args:
        data: Well preprocessed data in Pandas dataframe format.

    """
    create_dir('explore')
    stocks_data = data['stocks']
    weather_data = data['weather']
    stocks_close_data = pd.DataFrame().assign(Close=stocks_data['Close'])

    # Explore seasonal cycles using a 30-day rolling average
    season_explore(stocks_data.loc[:, 'Close'])

    # Explore dependency on year and month via carpet plot/heatmap
    year_month_dep_explore(stocks_close_data)

    # Explore dependency on day of the week and month via carpet plot/heatmap
    month_week_dep_explore(stocks_close_data)

    # Explore weekdays and weekends trends
    weekdays_weekends_explore(stocks_close_data)

    # explore feature correlation for each dataset separately
    feature_correlation_explore(stocks_data, 'stocks')
    feature_correlation_explore(weather_data, 'weather')

    # Explore the correlation between stock price data and other external data
    concatenated_data = pd.concat([stocks_data['Close'], weather_data], axis=1)
    correlation_explore(concatenated_data, 'sw')

    # hypothesis testing 1.defining hypothesis: sample two groups of data, one is 20 sampled close value when there
    # is no PRCP, the other is 20 sampled close value when there is
    hypo_testing(concatenated_data)
    # H0: suppose ud >= 0, not rejected


def season_explore(data):
    """Explore seasonality of the data and plot the results.

    Calculate average stocks close value for each month, given several years.

    Args:
        data: Pandas dataframe.

    """
    seasonal_cycle = data.rolling(window=30, center=True).mean().groupby(data.index.dayofyear).mean()
    q25 = data.rolling(window=30, center=True).mean().groupby(data.index.dayofyear).quantile(0.25)
    q75 = data.rolling(window=30, center=True).mean().groupby(data.index.dayofyear).quantile(0.75)
    mdays = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ndays_m = mdays.copy()
    ndays_m[2] = 29
    ndays_m = np.cumsum(ndays_m)
    ndays_m = ndays_m[:-1]
    month_ticks = month_abbr[1:]
    f, ax = plt.subplots(figsize=(10, 7))

    seasonal_cycle.plot(ax=ax, lw=2, color='b', legend=False)
    ax.fill_between(seasonal_cycle.index, q25.values.ravel(), q75.values.ravel(), color='b', alpha=0.3)
    ax.set_xticks(ndays_m + 15)
    ax.set_xticklabels(month_ticks)
    ax.grid(ls=':')
    ax.set_xlabel('Month', fontsize=15)
    ax.set_ylabel('Close value', fontsize=15)
    ax.set_xlim(0, 365)
    # ax.set_ylim(10000, 40000)
    [label.set_fontsize(13) for label in ax.xaxis.get_ticklabels()]
    [label.set_fontsize(13) for label in ax.yaxis.get_ticklabels()]

    ax.set_title('30 days stocks value', fontsize=15)
    plt.savefig('explore/season_cycle.png')
    plt.close()


def year_month_dep_explore(data):
    """Explore data dependencies between year and month for the stock data.

    Args:
        data: Pandas dataframe.

    """
    month_year = data.copy()
    month_year.loc[:, 'year'] = month_year.index.year
    month_year.loc[:, 'month'] = month_year.index.month
    month_year = month_year.groupby(['year', 'month']).mean().unstack()
    month_year.columns = month_year.columns.droplevel(0)
    f, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(month_year, ax=ax, cmap='Blues')

    cbax = f.axes[1]
    [label.set_fontsize(13) for label in cbax.yaxis.get_ticklabels()]
    cbax.set_ylabel('Stock Close Value', fontsize=13)

    [ax.axhline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 6)]
    [ax.axvline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 12)]

    ax.set_title('Stock Close Value per year and month', fontsize=16)

    [label.set_fontsize(13) for label in ax.xaxis.get_ticklabels()]
    [label.set_fontsize(13) for label in ax.yaxis.get_ticklabels()]

    ax.set_xlabel('Month', fontsize=15)
    ax.set_ylabel('Year', fontsize=15)
    ax.set_yticklabels(np.arange(2017, 2023, 1), rotation=0)
    plt.savefig('explore/year_month_dep.png')
    plt.close()


def month_week_dep_explore(data):
    """Explore data dependencies between month and week for the stock data.

    Args:
        data: Pandas dataframe.

    """
    month_day = data.copy()
    month_day.loc[:, 'day_of_week'] = month_day.index.dayofweek
    month_day.loc[:, 'month'] = month_day.index.month
    month_day = month_day.groupby(['day_of_week', 'month']).mean().unstack()
    month_day.columns = month_day.columns.droplevel(0)
    f, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(month_day, ax=ax, cmap='Greens')

    cbax = f.axes[1]
    [label.set_fontsize(13) for label in cbax.yaxis.get_ticklabels()]
    cbax.set_ylabel('Stock Close Value', fontsize=13)

    [ax.axhline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 7)]
    [ax.axvline(x, ls=':', lw=0.5, color='0.8') for x in np.arange(1, 12)]

    ax.set_title('Stock Close Value per day of the week and month', fontsize=16)

    [label.set_fontsize(13) for label in ax.xaxis.get_ticklabels()]
    [label.set_fontsize(13) for label in ax.yaxis.get_ticklabels()]

    ax.set_xlabel('Month', fontsize=15)
    ax.set_ylabel('Day of the week', fontsize=15)
    ax.set_yticklabels(day_abbr[0:7])
    plt.savefig('explore/month_week_dep.png')
    plt.close()


def weekdays_weekends_explore(data):
    """Explore data dependencies between weekdays and weekends for the stock data.

    Args:
        data: Pandas dataframe.

    """
    weekdays = data.loc[data.index.dayofweek.isin([0, 1, 2, 3, 4]), 'Close']
    weekends = data.loc[data.index.dayofweek.isin([5, 6]), 'Close']
    summary_month_weekdays = weekdays.groupby(weekdays.index.month).describe()
    summary_month_weekends = weekends.groupby(weekends.index.month).describe()
    f, ax = plt.subplots(figsize=(10, 7))

    ax.plot(summary_month_weekends.index, summary_month_weekends.loc[:, 'mean'], color='y', label='Weekends', ls='--',
            lw=3)
    ax.fill_between(summary_month_weekends.index, summary_month_weekends.loc[:, '25%'],
                    summary_month_weekends.loc[:, '75%'], facecolor='y', alpha=0.1)
    ax.plot(summary_month_weekdays.index, summary_month_weekdays.loc[:, 'mean'], color='b', label='Weekdays', lw=3)
    ax.fill_between(summary_month_weekdays.index, summary_month_weekdays.loc[:, '25%'],
                    summary_month_weekdays.loc[:, '75%'], facecolor='b', alpha=0.1)
    ax.legend(fontsize=15)
    ax.set_xticks(range(1, 13))
    ax.grid(ls=':', color='0.8')
    ax.set_xlabel('Month', fontsize=15)
    ax.set_ylabel('Stocks Close Value', fontsize=15)

    [label.set_fontsize(13) for label in ax.xaxis.get_ticklabels()]
    [label.set_fontsize(13) for label in ax.yaxis.get_ticklabels()]
    plt.savefig('explore/weekday_weekend_comp.png')
    plt.close()


def feature_correlation_explore(df, name):
    """Plot feature correlation matrix using pandas built-in functions for single dataset.

    Args:
        df: Pandas dataframe.
        name: Description of the specific dataset.

    """
    pd.plotting.scatter_matrix(df, range_padding=0.5, alpha=0.2)
    plt.savefig('explore/scm_' + name + '.png')
    plt.close()


def correlation_explore(df, name):
    """Explore feature correlation between stock data and auxiliary data.

    Args:
        df: Pandas dataframe.
        name: Description of the specific dataset.

    """
    col_names = list(df)
    plt.figure(figsize=[12.8, 4.8])
    plt.subplot(151)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[1])
    plt.subplot(152)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 2])
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[2])
    plt.subplot(153)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 3])
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[3])
    plt.subplot(154)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 4])
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[4])
    plt.subplot(155)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 5])
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[5])
    plt.subplots_adjust(wspace=1)
    plt.savefig('explore/scm_' + name + '.png')
    plt.close()


def check_normality(data):
    """Judge if the given data follows normal distribution.

    Args:
        data: Data in list form

    """
    test_stat_normality, p_value_normality = stats.shapiro(data)
    print("p value:%.4f" % p_value_normality)
    if p_value_normality < 0.05:
        print("Reject null hypothesis >> The data is not normally distributed")
    else:
        print("Fail to reject null hypothesis >> The data is normally distributed")


def hypo_testing(df):
    """Provide hypothesis testing.

    Judge whether stock close value is related to weather condition like precipitation.

    Args:
        df: Pandas dataframe.

    """
    values_no_prcp = []
    values_with_prcp = []
    for i in range(len(df.index)):
        cur_index = df.index[i]
        if df.loc[cur_index, 'PRCP'] == 0:
            values_no_prcp.append(df.loc[cur_index, 'Close'])
        else:
            values_with_prcp.append(df.loc[cur_index, 'Close'])
    values_no_prcp_sample = random.sample(values_no_prcp, 20)
    values_with_prcp_sample = random.sample(values_with_prcp, 20)

    check_normality(values_no_prcp_sample)
    check_normality(values_with_prcp_sample)

    test_stat, p_value_paired = stats.ttest_rel(values_no_prcp_sample, values_with_prcp_sample)
    print("p value:%.6f" % p_value_paired, "one tailed p value:%.6f" % (p_value_paired / 2))
    if p_value_paired < 0.05:
        print("Reject null hypothesis")
    else:
        print("Fail to reject null hypothesis")
