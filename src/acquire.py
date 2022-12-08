import json
import os

import yfinance as yf
import pandas as pd


def acquire(dataset_name):
    if dataset_name == 'stocks':
        return acquire_stock_data()


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

    current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(current_folder)
    data_filepath = os.path.join(current_folder, "data", "stock_data.json")
    print(data_filepath)
    # store dataset into json format on computer local disk
    with open(data_filepath, 'w') as outfile:
        outfile.write(aal_stock_data.to_json(orient='records'))

    return stock_data


if __name__ == "__main__":
    acquire_stock_data()
