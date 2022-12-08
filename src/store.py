import json
import os.path

import pymongo
from .acquire import acquire_stock_data
from .constants import MONGODB_SERVER_ADDRESS


# install missing library
# !pip install pymongo
# !pip install dnspython

# get mongo credentials from local file
def get_mongo_credentials(filepath):
    if filepath is None:
        current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(current_folder, "mongo_credential.json")
    with open(filepath, 'r') as credential_file:
        credential = json.load(credential_file)
        username = credential["username"]
        password = credential["password"]
    return username, password


def get_server(credentials_filepath=None):
    username, password = get_mongo_credentials(credentials_filepath)
    # connect to local mongo instance
    address = MONGODB_SERVER_ADDRESS.format(username=username, password=password)
    server = pymongo.MongoClient(address)
    return server


def store(data, db_name, col_name):
    server = get_server()
    db = server[db_name]
    cur_col = db[col_name]
    cur_col.insert_many(data)
    print("finish")


def contains_collection(db_name, col_name):
    server = get_server()

    # check if database exists
    database_list = server.list_database_names()
    if db_name not in database_list:
        return False

    # check if collection exists
    collection_list = server[db_name].list_collection_names()
    if col_name not in collection_list:
        return False

    # check if collection contains elements
    collection = server[db_name][col_name]
    if collection.count_documents({}) <= 0:
        return False

    return True


# testing
if __name__ == "__main__":
    store(acquire_stock_data())
