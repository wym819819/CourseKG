# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/database/__init__.py
# Description: 定义pdf文档解析器

from .base import *
from .parser import Parser, Page, Content, ContentType
import fitz
from paddleocr import PPStructure
from PIL import Image
import numpy as np
import cv2
import re


def _structure_result_sort(result: list[dict], epsilon: int = 5) -> list[dict]:
    """ 对版面分析的结果进行重排序，以符合阅读习惯

    Args:
        result (list[dict]): 版面分析结果
        epsilon (int, optional): 坐标误差容忍值. Defaults to 5.

    Returns:
        list[dict]: 重排序后的结果
    """
    # 按左上角的坐标排序
    # 首先按 y1 排序，再按 x1 排序
    result.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))

    sorted_result = []
    current_line = []
    last_y = None

    for item in result:
        x1, y1, x2, y2 = item['bbox']
        if last_y is None or abs(y1 - last_y) <= epsilon:
            current_line.append(item)
        else:
            # 对当前行按 x1 排序
            current_line.sort(key=lambda item: item['bbox'][0])
            sorted_result.extend(current_line)
            current_line = [item]
        last_y = y1

    if current_line:
        current_line.sort(key=lambda item: item['bbox'][0])
        sorted_result.extend(current_line)

    return sorted_result


def _replace_linefeed(sentence: str, ignore_end=True, replace='') -> str:
    """ 移除句子的换行符

    Args:
        sentence (str): 句子
        ignore_end (bool, optional): 忽略句末的换行符. Defaults to True.
        replace (str, optional): 换行符替换对象. Defaults to ''.

    Returns:
        str: 新句
    """
    if ignore_end:
        sentence_endings = r'[。！？.!?]'
        pattern = r'(?<!' + sentence_endings + r')\n'
    else:
        pattern = r'\n'
    sentence = re.sub(pattern, replace, sentence)
    return sentence


class PDFParser(Parser):

    def __init__(self, pdf_path: str) -> None:
        """ 解析pdf文档, 需要带有书签以判断层级

        Args:
            pdf_path (str): pdf文档路径
        """
        super().__init__(pdf_path)
        self.__pdf = fitz.open(pdf_path)
        self.__ocr_engine = None

    def __enter__(self) -> 'PDFParser':
        return self

    def close(self) -> None:
        """ 关闭文档
        """
        self.__pdf.close()

    def get_bookmarks(self) -> list[BookMark]:
        """  获取pdf文档书签

        Returns:
            list[BookMark]: 书签列表
        """
        stack: list[BookMark] = []
        bookmarks: list[BookMark] = []
        result = self.__pdf.get_toc()
        for item in result:
            level, title, page = item
            page -= 1  # 从0开始
            level -= 1  # 从0开始
            bookmarks.append(
                BookMark(
                    id='1:' + str(uuid.uuid4()) + f':{level}',
                    title=title,
                    page_index=page,
                    page_end=0,  # 结束页码需要由下一个书签确定
                    level=level,
                    subs=[]))

        for bookmark in reversed(bookmarks):
            level = bookmark.level

            while stack and stack[-1].level > level:
                bookmark.subs.append(stack.pop())

            stack.append(bookmark)

        stack.reverse()

        # 设置各个书签的结束页码
        def set_page_end(bookmarks: list[BookMark]):
            for idx in range(len(bookmarks)):
                if idx != len(bookmarks) - 1:
                    bookmarks[idx].set_page_end(bookmarks[idx + 1].page_index)
                set_page_end(bookmarks[idx].subs)

        set_page_end(stack)
        stack[-1].set_page_end(self.__pdf.page_count - 1)

        return stack

    def get_content(self, bookmark: BookMark) -> list[Content]:
        """  获取书签下的所有内容

        Args:
            bookmark (BookMark): 书签

        Returns:
            list[Content]: 内容列表
        """
        # 获取书签对应的页面内容
        contents: list[Content] = []
        # 后续这个地方可以并行执行
        for pg in range(bookmark.page_index, bookmark.page_end + 1):

            # 起始页内容定位
            page_contents = self.get_page(pg).contents
            if pg == bookmark.page_index:
                idx = 0
                for i, content in enumerate(page_contents):
                    blank_pattern = re.compile(r'\s+')  # 可能会包含一些空白字符这里去掉
                    if content.type == ContentType.Title and re.sub(
                            blank_pattern, '', content.content) == re.sub(
                                blank_pattern, '', bookmark.title):
                        idx = i + 1
                        break
                page_contents = page_contents[idx:]
            # 终止页内容定位
            if pg == bookmark.page_end:
                idx = len(page_contents)
                for i, content in enumerate(page_contents):
                    if content.type == ContentType.Title:  # 直到遇到下一个标题为止，这里的逻辑可能存在问题~
                        idx = i
                        break
                page_contents = page_contents[:idx]
            contents.extend(page_contents)
        return contents

    def get_page(self, page_index: int, structure=True) -> Page:
        """ 获取文档页面, 可以使用版面分析获得更好的内容结构

        Args:
            page_index (int): 页码, 从0开始计数
            structure (bool, optional): 使用 PP-Structure 进行版面分析. Defaults to True.

        Returns:
            Page: 文档页面
        """
        if structure:
            if self.__ocr_engine is None:
                self.__ocr_engine = PPStructure(table=False,
                                                ocr=True,
                                                show_log=False)
            pdf_page = self.__pdf[page_index]
            # 不需要对页面进行缩放
            mat = fitz.Matrix(1, 1)
            pm = pdf_page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            result: list[dict] = self.__ocr_engine(img)
            page = Page(page_index=page_index, contents=[])
            result = _structure_result_sort(result)
            for item in result:
                content = pdf_page.get_textbox(item['bbox'])
                if item['type'] == 'text':
                    content = _replace_linefeed(content)
                    page.contents.append(
                        Content(type=ContentType.Text, content=content))
                elif item['type'] == 'title':
                    page.contents.append(
                        Content(type=ContentType.Title, content=content))
            return page
        else:
            pdf_page = self.__pdf[page_index]
            return Page(page_index=page_index + 1,
                        contents=[
                            Content(type=ContentType.Text,
                                    content=pdf_page.get_text())
                        ])

    def get_pages(self, structure=True) -> list[Page]:
        """ 获取pdf文档所有页面, 可以使用版面分析获得更好的内容结构

        Args:
            structure (bool, optional): 使用 PP-Structure 进行版面分析. Defaults to True.

        Returns:
            list[Page]: 页面列表
        """
        pages: list[Page] = []
        for pg in range(0, self.__pdf.page_count):
            pages.append(self.get_page(page_index=pg, structure=structure))
        return pages
