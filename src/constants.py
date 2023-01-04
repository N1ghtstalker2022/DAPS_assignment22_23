"""Summarise constants

This python file includes constants that can be used by the whole project

"""
from enum import Enum

MONGODB_SERVER_ADDRESS = (
    "mongodb+srv://Eden:4FKm7FYMc18pcaXY@cluster0.ytnbwiv.mongodb.net/")

DATABASE_NAME = "daps_data"

Collections = Enum('Collections', ('stocks', 'weather', 'covid'))

