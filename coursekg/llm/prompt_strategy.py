# -*- coding: utf-8 -*-
# Create Date: 2024/07/13
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/llm/prompt_strategy.py
# Description: 定义提示词示例检索策略

from ..database import Mongo, Faiss
from pymongo.collation import Collation
import os
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from glob import glob


class Database:

    def __init__(self, faiss: Faiss, mongo: Collation) -> None:
        """ 数据库组合类

        Args:
            faiss (Faiss): faiss 向量数据库
            mongo (Collation): mongo collection 文档数据库
        """
        self.faiss = faiss
        self.mongo = mongo


class ExamplePromptStrategy:

    def __init__(self,
                 embed_model_path: str,
                 mongo_url: str = 'mongodb://localhost:27017/',
                 faiss_path: str = 'coursekg/database/faiss_index') -> None:
        """ 提示词示例检索策略

        Args:
            embed_model_path (str): 嵌入模型路径
            mongo_url (str, optional): 文档数据库 mongodb 地址. Defaults to 'mongodb://localhost:27017/'.
            faiss_path (str, optional): 向量数据库 faiss 存储地址. Defaults to 'coursekg/database/faiss_index'.
        """
        mongo = Mongo(mongo_url, 'coursekg')
        self.db_ner = Database(
            faiss=Faiss(os.path.join(faiss_path, 'faiss_index_ner.bin')),
            mongo=mongo.get_collection('prompt_example_ner'))
        self.db_re = Database(faiss=Faiss(
            os.path.join(faiss_path, 'faiss_index_re.bin')),
                              mongo=mongo.get_collection('prompt_example_re'))
        self.embed_model = SentenceTransformer(embed_model_path)

    def reimport_example(
            self,
            embed_dim: int,
            example_dataset_path: str = 'dataset/prompt_example') -> None:
        """ 重新向数据库(文档数据库/向量数据库)中导入提示词示例

        Args:
            embed_dim (int): 嵌入维度
            example_dataset_path (str, optional): 提示词示例源数据地址文件夹. Defaults to 'dataset/prompt_example'.
        """

        for file in glob(example_dataset_path + '/*'):
            if file.endswith('ner.json'):
                db = self.db_ner
            elif file.endswith('re.json'):
                db = self.db_re
            else:
                continue
            # 清除文档数据库中已保存内容
            db.mongo.drop()
            examples = []
            idx = 0
            with open(file, 'r', encoding='UTF-8') as f:
                for line in json.load(f):
                    line['index'] = idx
                    idx += 1
                    examples.append(line)
            db.mongo.insert_many(examples)
            data = []
            for line in examples:
                data.append(
                    self.embed_model.encode(line['input'],
                                            normalize_embeddings=True))
            # 清除向量数据库中已保存内容
            db.faiss.delete()
            db.faiss.create(embed_dim).add(
                np.array(data).astype('float32')).save()

    def get_ner_example(self, content: str) -> list:
        """ 获取实体抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Returns:
            list: 提示词示例列表
        """
        if self.db_ner.faiss.index is None:
            self.db_ner.faiss.load()
        content_vec = self.embed_model.encode(content,
                                              normalize_embeddings=True)
        idx = self.db_ner.faiss.search(
            np.array([content_vec]).astype('float32'), 3)
        examples = []
        for i in idx:
            res = self.db_ner.mongo.find_one({'index': i})
            examples.append({'input': res['input'], 'output': res['output']})
        return examples

    def get_re_example(self, content: str) -> list:
        """ 获取关系抽取提示词示例

        Args:
            content (str): 待抽取的文本内容

        Returns:
            list: 提示词示例列表
        """
        if self.db_re.faiss.index is None:
            self.db_re.faiss.load()
        content_vec = self.embed_model.encode(content,
                                              normalize_embeddings=True)
        idx = self.db_re.faiss.search(
            np.array([content_vec]).astype('float32'), 3)
        examples = []
        for i in idx:
            res = self.db_re.mongo.find_one({'index': i})
            examples.append({'input': res['input'], 'output': res['output']})
        return examples
