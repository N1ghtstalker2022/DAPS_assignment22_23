import yfinance as yf
import pandas as pd


def acquire():
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
        print(new_date)
        aal_stock_data.at[i, 'Date'] = 0
        aal_stock_data.at[i, 'Date'] = new_date
        print(aal_stock_data.at[i, 'Date'])

    # convert to json before storing into database
    stock_data = aal_stock_data.to_json(orient='records')

    # store dataset into csv format on computer local disk
    aal_stock_data.to_csv('aal_stock_data.csv', index=False, header=True)
    return stock_data
