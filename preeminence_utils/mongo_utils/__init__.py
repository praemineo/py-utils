"""
Centralised utility class for all operations to be performed on a mongo database.
"""
from pymongo import MongoClient

class MongoUtils:
  def __init__(self,address="127.0.0.1", port=27017, db_name="", collection_name=""):
    """
    Initialise collection object
    :param address: database ip
    :param port: mongo port
    :param db_name: Databse name
    :param collection_name: Collection name
    """
    self.address = address
    self.port = port
    self.db_name = db_name
    self.collection_name = collection_name
    self.collection = self.connect_to_db()


  def update_record(self, filter_condition={}, new_value={}):
    """
    Connect to mongo db and update data.

    :param filter_condition: Filter condition
    :param new_value: new value of the selected document
    :return: Return collection as a list
    """

    self.collection.update_one(filter_condition, {"$set": new_value})

  def insert_record(self,new_record={}):
    """
    Insert a record in table
    :param new_record: New record to be inserted.
    :return:
    """
    self.collection.insert_one(new_record)

  def get_list_from_db(self, filter_condition={}):
    """
    Connect to mongo db and fetch data.

    :param filter_condition: Filter condition
    :return: Return collection as a list
    """
    return list(self.collection.find(filter_condition))

  def connect_to_db(self):
    client = MongoClient(self.address, self.port)
    db = client[self.db_name]
    collection = db[self.collection_name]
    return collection