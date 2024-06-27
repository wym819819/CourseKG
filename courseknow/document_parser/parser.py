from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Document, BookMark


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
class Page:
    """ 页面
    """
    page_index: int
    contents: list[Content]


class Parser(ABC):

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

    @abstractmethod
    def close(self) -> None:
        """ 关闭文档
        """
        raise NotImplementedError

    @abstractmethod
    def get_bookmarks(self) -> list[BookMark]:
        """  获取pdf文档书签

        Returns:
            list[BookMark]: 书签列表
        """
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def get_pages(self) -> list[Page]:
        """ 获取文档所有页面

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list[Page]: 页面列表
        """
        raise NotImplementedError

    @abstractmethod
    def get_document(self) -> Document:
        """ 获取文档

        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            Document: 文档
        """
        raise NotImplementedError
