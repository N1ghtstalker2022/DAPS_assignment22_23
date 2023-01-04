"""This python file is the entrance of the program.

It follows the procedures displayed below.
1. Acquire stock data.
2. Collect auxiliary data that might have an impact on the company’s stocks.
3. Choose the storing strategy that most efficiently supports the upcoming data analysis.
4. Check for any missing/noisy/outlier data, and clean it, only if necessary.
5. Process the data, extracting features that you believe are meaningful to forecast the trend
of the stock.
6. Provide exploratory data analysis and generate useful visualisations of the data.
7. Train a model to predict the closing stock price.

"""
from src import *


def main():
    """This is the start point of the whole project.

    This function performs data acquisition followed by data storage, data preprocessing, data exploration and data
    inference.

    """
    # get database name, assume only one database is used
    db_name = constants.DATABASE_NAME

    # get collection names
    col_name_enum = constants.Collections

    preprocessed_data = {}
    for col_name, _ in col_name_enum.__members__.items():
        # data acquisition
        cur_data = acquire.acquire(col_name)
        # data storing into local disk
        store.store_local(cur_data, col_name + '_data.json')

        if not store.contains_collection(db_name, col_name):
            # data storing into cloud-based database
            store.store_cloud(cur_data, db_name, col_name)

        # start data preprocessing
        # raw_data = store.read_from_db(db_name, col_name)
        data = preprocess.preprocess(col_name)

        preprocessed_data[col_name] = data

    # data exploration
    explore.explore(preprocessed_data)
    inference.infer(preprocessed_data)

    print("...")


if __name__ == "__main__":
    main()
