from dataclasses import dataclass
from enum import Enum
import uuid


class ContentType(Enum):
    """ 内容类型
    """
    Text = 1
    Title = 2


@dataclass
class Content:
    """ 内容
    """
    type: ContentType
    content: str


@dataclass
class KonwledgePoint:
    """ 知识点
    """
    id: str
    name: str


@dataclass
class BookMark:
    """ 书签
    """
    id: str
    title: str
    page_index: int
    page_end: int
    level: int
    subs: list['BookMark'] | list[KonwledgePoint] | None

    def set_page_end(self, page_end: int) -> None:
        """ 设置书签的结束页, 和直接修改 BookMark 对象的 page_end 属性不同, 该方法会考虑到书签嵌套的情况

        Args:
            page_end (int): 结束页码
        """
        self.page_end = page_end
        if self.subs and isinstance(self.subs[-1], BookMark):
            self.subs[-1].set_page_end(page_end)


@dataclass
class Document:
    """文档
    """
    id: str
    name: str
    bookmarks: list[BookMark]


@dataclass
class Page:
    """ 页面
    """
    page_index: int
    contents: list[Content]


class Parser:

    def __init__(self, file_path: str) -> None:
        """ 文档解析器基类

        Args:
            file_path (str): 文档路径
        """
        self.file_path = file_path

    def __enter__(self) -> 'Parser':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """ 关闭文档
        """
        raise NotImplementedError

    def get_bookmarks(self) -> list[BookMark]:
        """  获取pdf文档书签

        Returns:
            list[BookMark]: 书签列表
        """
        raise NotImplementedError

    def get_page(self, page_index: int) -> Page:
        """ 获取文档页面

        Args:
            page_index (int): 页码, 从0开始计数

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            Page: 文档页面
        """
        raise NotImplementedError

    def get_pages(self) -> list[Page]:
        """ 获取文档所有页面

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list[Page]: 页面列表
        """
        raise NotImplementedError

    def get_document(self) -> Document:
        """ 获取文档

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            Document: 文档
        """
        raise NotImplementedError
