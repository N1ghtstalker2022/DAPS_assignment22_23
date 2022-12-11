import json
import os

import yfinance as yf
import pandas as pd
import requests


def acquire(dataset_name):
    if dataset_name == 'stocks':
        return acquire_stock_data()
    elif dataset_name == 'weather':
        return acquire_weather_data()


def acquire_stock_data():
    # choose American Airline Group
    aal_stock = yf.Ticker("AAL")

    # find stock history with one day interval
    aal_stock_data = aal_stock.history(start="2017-04-01", end="2022-04-01", interval="1d")

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
    weather_url_ny = "https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&startDate=2017-04-01" \
                     "&endDate=2022-04-01&format=json&includeStationName=false&includeStationLocation=0&stations" \
                     "=US1NYWC0018 "
    weather_ny = requests.get(weather_url_ny).json()
    return weather_ny


if __name__ == "__main__":
    acquire_weather_data()
