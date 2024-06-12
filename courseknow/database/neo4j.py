from py2neo import Graph
from tqdm.rich import tqdm


def singleton(cls):

    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class Neo4j:

    def __init__(self, url: str, username: str, password: str) -> None:
        """ 连接到 Neo4j 数据库

        Args:
            url (str): 地址
            username (str): 用户名
            password (str): 密码
        """
        self.graph = Graph(url, auth=(username, password), name='neo4j')

    def run(self, cyphers: str | list[str]) -> None:
        """ 执行一条或多条cypher语句

        Args:
            cyphers (str | list[str]): 一条或多条cypher语句
        """
        if isinstance(cyphers, str):
            self.graph.run(cyphers)
        else:
            for cypher in tqdm(cyphers):
                self.graph.run(cypher)
