# -*- coding: utf-8 -*-
# Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File: coursekg/database/mongo.py
# Description: 定义 mongodb 数据库连接类

from pymongo import MongoClient
from .singleton import singleton


@singleton
class Mongo:

    def __init__(self, url: str, db: str, collection: str) -> None:
        """ 连接到 Mongodb 数据库

        Args:
            url (str): 地址
            db (str): 数据库
            collection (str): 集合
        """
        self.__client = MongoClient(url)
        self.__db = self.__client[db]
        self.collection = self.__db[collection]
