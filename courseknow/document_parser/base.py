from dataclasses import dataclass
from enum import Enum
from ..llm import LLM, KnowledgepointPrompt
import uuid
import re
from loguru import logger
from sentence_transformers import SentenceTransformer
import os
from ..util import get_root_path

logger.remove(0)
logger.add(os.path.join(get_root_path(), "log/parser.log"),
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
           mode="w")


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
class KnowledgePoint:
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
    subs: list['BookMark'] | list[KnowledgePoint] | None

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
    parser: 'Parser'

    def set_knowledgepoints_by_llm(self, llm: LLM,
                                   prompt: KnowledgepointPrompt) -> None:
        """ 使用 LLM 抽取知识点存储到 BookMark 中

        Args:
            llm (LLM): 指定 LLM
            domain (str): 知识点相关的领域
        """
        points: list[KnowledgePoint] = []

        # 使用句嵌入模型比较知识点相关性
        threshold = 0.9
        model = SentenceTransformer(
            os.path.join(get_root_path(),
                         'pretrain/iampanda/zpoint_large_embedding_zh'))

        def set_knowledgepoints(bookmarks: list[BookMark]) -> None:
            for bookmark in bookmarks:
                if bookmark.subs and isinstance(bookmark.subs[-1], BookMark):
                    set_knowledgepoints(bookmark.subs)
                else:
                    logger.success('子章节: ' + bookmark.title)
                    # 获取书签对应的页面内容
                    contents: list[Content] = []
                    for pg in range(bookmark.page_index,
                                    bookmark.page_end + 1):

                        # 起始页和终止页需要进行内容定位
                        page_contents = self.parser.get_page(pg).contents
                        if pg == bookmark.page_index:
                            idx = 0
                            for i, content in enumerate(page_contents):
                                blank_pattern = re.compile(r'\s+')
                                if content.type == ContentType.Title and re.sub(
                                        blank_pattern, '',
                                        content.content) == re.sub(
                                            blank_pattern, '', bookmark.title):
                                    idx = i + 1
                                    break
                            page_contents = page_contents[idx:]
                        if pg == bookmark.page_end:
                            idx = len(page_contents)
                            for i, content in enumerate(page_contents):
                                if content.type == ContentType.Title:  # 直到遇到下一个标题为止
                                    idx = i
                                    break
                            page_contents = page_contents[:idx]
                        contents.extend(page_contents)
                    text_contents = '\n'.join(
                        [content.content for content in contents])
                    text_contents = text_contents.strip()
                    if len(text_contents) == 0:
                        bookmark.subs = []
                        continue
                    resp = llm.chat(prompt.get_prompt(text_contents))
                    logger.success('模型返回: ' + resp)
                    generate_points = prompt.post_process(resp)
                    logger.success('生成知识点: ' + str(generate_points))
                    subs: list[KnowledgePoint] = []
                    for generate in generate_points:
                        generate_name_embed = model.encode(
                            generate, normalize_embeddings=True)
                        for point in points:
                            if generate == point.name or (  # 名称相同则无需计算
                                    similarity :=
                                    generate_name_embed @ point.name_embed
                            ) > threshold:  # 实体归一化 判断余弦相似度
                                logger.info(f'{generate} {point.name} ' +
                                            ('相同' if generate ==
                                             point.name else f'{similarity}'))
                                subs.append(point)
                                break
                        else:
                            new_point = KnowledgePoint(id='2:' +
                                                       str(uuid.uuid4()),
                                                       name=generate)
                            # 额外设置嵌入向量属性
                            setattr(new_point, 'name_embed',
                                    generate_name_embed)
                            subs.append(new_point)
                            points.append(new_point)
                    # subs需要去重 使用name即可
                    unique_objs_dict = {obj.name: obj for obj in subs}
                    subs = list(unique_objs_dict.values())
                    bookmark.subs = subs

        set_knowledgepoints(self.bookmarks)

    def get_cyphers(self) -> list[str]:
        """ 将章节与知识点之间的关联关系存入图数据库中

        Returns:
            list[str]: 多条 cypher 语句
        """

        created_knowledgepoint_ids = []
        cypher = f'CREATE (:Document {{id: "{self.id}", name: "{self.name}"}})'

        def bookmarks_to_cypher(bookmarks: list[BookMark], parent_id: str):
            cyphers: list[str] = []
            for bookmark in bookmarks:
                cyphers.append(
                    f'CREATE (:Chapter {{id: "{bookmark.id}", name: "{bookmark.title}", page_start: {bookmark.page_index}, page_end: {bookmark.page_end}}})'
                )
                cyphers.append(  # 这里写不写类别无所谓
                    f'MATCH (n1 {{id: "{parent_id}"}}) MATCH (n2:Chapter {{id: "{bookmark.id}"}}) CREATE (n1)-[:SubChapter {{name: "子章节"}}]->(n2)'
                )
                if bookmark.subs and isinstance(bookmark.subs[-1], BookMark):
                    cyphers.extend(
                        bookmarks_to_cypher(bookmark.subs, bookmark.id))
                elif bookmark.subs and isinstance(bookmark.subs[-1],
                                                  KnowledgePoint):
                    for point in bookmark.subs:
                        if point.id in created_knowledgepoint_ids:
                            cyphers.append(
                                f'MATCH (n1:Chapter {{id: "{bookmark.id}"}}) MATCH (n2:KnowledgePoint {{id: "{point.id}"}}) CREATE (n1)-[:Has {{name: "提到知识点"}}]->(n2)'
                            )
                        else:
                            cyphers.append(
                                f'MATCH (n1:Chapter {{id: "{bookmark.id}"}}) CREATE (n2:KnowledgePoint {{id: "{point.id}", name: "{point.name}"}}) CREATE (n1)-[:Has {{name: "提到知识点"}}]->(n2)'
                            )
                            created_knowledgepoint_ids.append(point.id)
            return cyphers

        cyphers = [cypher]
        cyphers.extend(bookmarks_to_cypher(self.bookmarks, self.id))
        return cyphers


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
