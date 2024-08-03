# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/document_parser/resource.py
# Description: 定义资源类

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from pptx import Presentation
from ..llm import LLM, Prompt
from collections import Counter


@dataclass
class Slice:
    """ 资源切片
    """
    file_path: str
    start: int | str
    end: int | str

    def __repr__(self) -> str:
        return f"Slice<path={self.file_path}, range=[{self.start},{self.end}]>"


class Resource(ABC):

    def __init__(self, file_path: str) -> None:
        """ 资源基类

        Args:
            file_path (str): 资源文件路径
        """
        super().__init__()
        self.file_path = file_path
        self.slices_maps: dict[str, list[Slice]] = dict()

    def __repr__(self) -> str:
        return f"Resource<path={self.file_path}>"

    def get_slices(self, keyword: str) -> list[Slice]:
        """ 通过关键词获取切片

        Args:
            keyword (str): 关键词

        Returns:
            list[Slice]: 切片列表
        """
        return self.slices_maps.get(keyword, None)


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

    def __getstate__(self):
        """ 自定义序列化方法
        """
        state = self.__dict__.copy()
        # 移除 parser 属性
        del state['pptx']
        return state

    def __setstate__(self, state):
        """ 自定义反序列化方法
        """
        self.__dict__.update(state)
        self.pptx = Presentation(state['file_path'])


@dataclass
class ResourceMap:
    """ 资源和书签的关联关系
    """
    bookmark_title: str
    resource: Resource
