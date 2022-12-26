import random

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def explore(data):
    stocks_data = data['stocks']
    weather_data = data['weather']
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


def feature_correlation_explore(df, name):
    fig = pd.plotting.scatter_matrix(df, range_padding=0.5, alpha=0.2)
    plt.savefig('demo/scm_' + name + '.png')
    plt.close()


def correlation_explore(df, name):
    col_names = list(df)
    figure = plt.figure(figsize=[12.8, 4.8])
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
    plt.savefig('demo/scm_' + name + '.png')
    plt.close()


def check_normality(data):
    test_stat_normality, p_value_normality = stats.shapiro(data)
    print("p value:%.4f" % p_value_normality)
    if p_value_normality < 0.05:
        print("Reject null hypothesis >> The data is not normally distributed")
    else:
        print("Fail to reject null hypothesis >> The data is normally distributed")


def hypo_testing(df):
    values_noPRCP = []
    values_withPRCP = []
    for i in range(len(df.index)):
        cur_index = df.index[i]
        if df.loc[cur_index, 'PRCP'] == 0:
            values_noPRCP.append(df.loc[cur_index, 'Close'])
        else:
            values_withPRCP.append(df.loc[cur_index, 'Close'])
    values_noPRCP_sample = random.sample(values_noPRCP, 20)
    values_withPRCP_sample = random.sample(values_withPRCP, 20)

    check_normality(values_noPRCP_sample)
    check_normality(values_withPRCP_sample)

    test_stat, p_value_paired = stats.ttest_rel(values_noPRCP_sample, values_withPRCP_sample)
    print("p value:%.6f" % p_value_paired, "one tailed p value:%.6f" % (p_value_paired / 2))
    if p_value_paired < 0.05:
        print("Reject null hypothesis")
    else:
        print("Fail to reject null hypothesis")
