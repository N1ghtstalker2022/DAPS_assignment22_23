"""Data storage procedure.

This python file stores acquired data into local disks and cloud-based mongodb separately.

Typical usage example:

    store_local('data', 'file_name')
    store_cloud('data', 'database_name', 'collection_name')
"""
import json
import os.path
import pymongo
from .constants import MONGODB_SERVER_ADDRESS
from .utils import create_dir


def get_mongo_credentials(filepath):
    """Get personal credentials to connect with mongodb.

    Args:
        filepath: Filepath that stores mongodb credential.

    Returns:
        String formatted username and password

    """
    if filepath is None:
        current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(current_folder, "mongo_credential.json")
    with open(filepath, 'r') as credential_file:
        credential = json.load(credential_file)
        username = credential["username"]
        password = credential["password"]
    return username, password


def get_server(credentials_filepath=None):
    """Try to connect with remote mongodb database server

    Args:
        credentials_filepath: Filepath that stores mongodb credential.

    Returns:
        Python object that represents mongodb server.

    """
    username, password = get_mongo_credentials(credentials_filepath)
    # connect to local mongo instance
    address = MONGODB_SERVER_ADDRESS.format(username=username, password=password)
    server = pymongo.MongoClient(address)
    return server


def store_local(data, file_name):
    """Store data into local file.

    Args:
        data: Data in json list format.
        file_name: Local file name.

    """
    create_dir('data')
    current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_filepath = os.path.join(current_folder, "data", file_name)
    # store dataset into json format on computer local disk
    with open(data_filepath, 'w') as outfile:
        json.dump(data, outfile)


def store_cloud(data, db_name, col_name):
    """Store data into cloud database.

    Args:
        data: Data in json list format.
        db_name: Database name.
        col_name: Collection name.

    """
    server = get_server()
    db = server[db_name]
    cur_col = db[col_name]
    cur_col.insert_many(data)
    print("finish storing")


def contains_collection(db_name, col_name):
    """Check if collection already exists.

    To prevent duplicated documents.

    Args:
        db_name: Database name.
        col_name: Collection name.

    Returns:
        True for not existing collection.
        False for existing collection.

    """
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


def read_from_db(db_name, col_name):
    server = get_server()
    db = server[db_name]
    cur_col = db[col_name]
    all_data = cur_col.find()
    return all_data
