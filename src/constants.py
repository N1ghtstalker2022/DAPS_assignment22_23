from enum import Enum

MONGODB_SERVER_ADDRESS = (
    "mongodb+srv://Eden:4FKm7FYMc18pcaXY@cluster0.ytnbwiv.mongodb.net/")

DATABASE_NAME = "daps_data"

Collections = Enum('Collections', 'stocks')
for name, member in Collections.__members__.items():
    print(name, '=>', member, ',', member.value)
