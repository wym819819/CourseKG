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
from ..llm import LLM, Prompt
from collections import Counter


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

    def set_slices_by_llm(self,
                          llm: LLM,
                          prompt: Prompt,
                          samples: int = 3,
                          top: float = 0.5) -> None:
        """ 为知识点设置相应的资源

        Args:
            llm (LLM): 指定LLM
            prompt (Prompt): 使用的提示词 (建议复用知识抽取的提示词, 只需使用到其中的实体识别部分)
            samples (int, optional): 采用自我一致性策略的采样次数. Defaults to 5.
            top (float, optional): 采用自我一致性策略时，出现次数超过 top * samples 时才会被采纳，范围为 [0, 1]. Defaults to 0.5.
        """

        name_maps: dict[str, list[int]] = dict()
        for idx, slide in enumerate(self.pptx.slides):
            for shape in slide.shapes:
                if shape.has_text_frame:
                    # pptx当前页面的文字
                    text = shape.text_frame.text.strip()
                    text = text.strip()
                    if len(text) <= 10:
                        continue
                    all_name: list[str] = []
                    for idx in range(samples):
                        resp = llm.chat(prompt.get_ner_prompt(text))
                        entities_name = prompt.post_process(resp)
                        if isinstance(entities_name, dict):  # 解析失败或者没有产生结果
                            continue
                        all_name.extend(entities_name)
                    counter = Counter(all_name)
                    names = [
                        point for point, count in counter.items()
                        if count > (samples * top)
                    ]
                    for name in names:
                        if name in name_maps.keys():
                            name_maps[name].append(idx + 1)
                        else:
                            name_maps[name] = [idx]
        for name in name_maps:
            name_maps[name].sort()
            name_maps[name] = _merge_index_slice(name_maps[name],
                                                 self.file_path)
        self.slices_maps = name_maps


@dataclass
class ResourceMap:
    """ 资源和书签的关联关系
    """
    bookmark_title: str
    resource: Resource
