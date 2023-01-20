"""Data acquisition procedure.

This python file aims at acquiring stocks data for American Airlines Group Inc and auxiliary data including weather
data and covid-19 data in the USA.

Typical usage example:

    data = acquire('stocks')
"""
import yfinance as yf
import pandas as pd
import requests


def acquire(dataset_name):
    """Acquire dataset.

    Acquire dataset according to given dataset name.

    """
    if dataset_name == 'stocks':
        return acquire_stock_data()
    elif dataset_name == 'weather':
        return acquire_weather_data()
    elif dataset_name == 'covid':
        return acquire_covid_data()


def acquire_stock_data():
    """Acquire stocks data.

    Acquire stocks data by using Yahoo! Finance API.

    Returns:
        A python list containing stocks data from 2017-04-01 to 2022-04-30

    """
    # choose American Airline Group
    aal_stock = yf.Ticker("AAL")

    # find stock history with one day interval
    aal_stock_data = aal_stock.history(start="2017-04-01", end="2022-05-01", interval="1d")

    aal_stock_data = aal_stock_data.reset_index()

    # change date format
    for i in range(len(aal_stock_data)):
        date = aal_stock_data.at[i, "Date"]
        ts = pd.to_datetime(date)
        new_date = ts.strftime('%Y-%m-%d')
        aal_stock_data.at[i, 'Date'] = 0
        aal_stock_data.at[i, 'Date'] = new_date

    # convert to list
    stock_data = aal_stock_data.to_dict('records')

    return stock_data


def acquire_weather_data():
    """Acquire weather data.

    Acquire weather data for New York by using NOAA API.

    Returns:
        A python list containing weather data from 2017-04-01 to 2022-04-30

    """
    weather_url_ny = "https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&startDate=2017-04-01" \
                     "&endDate=2022-05-01&format=json&includeStationName=false&includeStationLocation=0&stations" \
                     "=US1NYWC0018 "
    weather_ny = requests.get(weather_url_ny).json()
    return weather_ny


def acquire_covid_data():
    """Acquire weather data.

    Acquire covid data for New York by using cdc API.

    Returns:
        A python list containing covid data up to 2022-04-30

    """
    covid_url = "https://data.cdc.gov/resource/9mfq-cb36.json?$where=submission_date<'2022-05-01T00:00:00.000' AND " \
                "state='NY' "
    covid_json = requests.get(covid_url).json()
    covid_df = pd.DataFrame.from_records(covid_json)
    # change date format
    for i in range(len(covid_df)):
        date = covid_df.at[i, "submission_date"]
        ts = pd.to_datetime(date)
        new_date = ts.strftime('%Y-%m-%d')
        covid_df.at[i, 'submission_date'] = 0
        covid_df.at[i, 'submission_date'] = new_date

    # convert to list
    covid_json = covid_df.to_dict('records')
    return covid_json
