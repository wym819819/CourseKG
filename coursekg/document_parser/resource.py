# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/document_parser/resource.py
# Description: 定义资源类

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pptx import Presentation
from .base import BookMark


@dataclass
class Slice:
    """ 资源切片
    """
    file_path: str
    start: int | str
    end: int | str


class Resource(ABC):

    def __init__(self, file_path: str) -> None:
        """ 资源基类

        Args:
            file_path (str): 资源文件路径
        """
        super().__init__()
        self.file_path = file_path

    def __repr__(self) -> str:
        return f"Resource<path={self.file_path}>"

    @abstractmethod
    def get_slices(self, keyword: str) -> list[Slice]:
        """ 通过关键词获取切片

        Args:
            keyword (str): 关键词
        
        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            list[Slice]: 切片列表
        """
        raise NotImplementedError


def _merge_index_slice(items: list[int], file_path: str) -> list[Slice]:
    """ 将连续的页数索引合并

    Args:
        items (list[int]): 页数列表
        file_path (str): 文件路径

    Returns:
        list[Slice]: 切片列表
    """
    if not items:
        return []

    items.sort()
    slices = []
    start = items[0]
    end = items[0]

    for i in range(1, len(items)):
        if items[i] == end + 1:
            end = items[i]
        else:
            slices.append(Slice(start=start, end=end, file_path=file_path))
            start = items[i]
            end = items[i]

    slices.append(Slice(start=start, end=end, file_path=file_path))
    return slices


class PPTX(Resource):

    def __init__(self, pptx_path: str) -> None:
        """ .pptx类型文件资源

        Args:
            pptx_path (str): 文件路径
        """
        super().__init__(pptx_path)
        self.pptx = Presentation(pptx_path)

    def get_slices(self, keyword: str) -> list[Slice]:
        """ 通过关键词获取切片

        Args:
            keyword (str): 关键词

        Returns:
            list[Slice]: 切片列表
        """
        idxs = []
        for idx, slide in enumerate(self.pptx.slides):
            for shape in slide.shapes:
                if shape.has_text_frame:
                    if (text := shape.text_frame.text.strip()
                        ) != '' and keyword in text:
                        idxs.append(idx + 1)  # 页数从1开始计算
                        break
        return _merge_index_slice(idxs, self.file_path)


@dataclass
class ResourceMap:
    """ 资源和书签的关联关系
    """
    bookmark_title: str
    resource: Resource
