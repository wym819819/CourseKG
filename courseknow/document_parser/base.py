from dataclasses import dataclass
from ..llm import LLM, Prompt
import uuid
import re
from loguru import logger
from .config import ignore_page, parser_log
import random
from collections import Counter
from .parser import Content, ContentType, Parser

logger.remove(0)
logger.add(parser_log,
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
           mode="w")


@dataclass
class KPEntity:
    """ 知识点实体
    """
    id: str
    name: str

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KPEntity):
            return self.id == other.id
        return False


@dataclass
class KPRelation:
    """ 知识点关系
    """
    id: str
    type_name: str


@dataclass
class KPTriple:
    """ 知识点三元组
    """
    head: KPEntity
    relation: KPRelation
    tail: KPRelation


@dataclass
class BookMark:
    """ 书签
    """
    id: str
    title: str
    page_index: int
    page_end: int
    level: int
    subs: list['BookMark'] | list[KPTriple] | None

    def set_page_end(self, page_end: int) -> None:
        """ 设置书签的结束页, 和直接修改 BookMark 对象的 page_end 属性不同, 该方法会考虑到书签嵌套的情况

        Args:
            page_end (int): 结束页码
        """
        self.page_end = page_end
        if self.subs and isinstance(self.subs[-1], BookMark):
            self.subs[-1].set_page_end(page_end)

    def to_dict(self) -> dict:
        """ 将 BookMark 对象转换为 dict 形式

        Returns:
            dict: dict 形式
        """
        return {
            'title':
            self.title,
            'page_range':
            f'[{self.page_index}, {self.page_end}]',
            'subs': [sub.to_dict()
                     for sub in self.subs] if self.subs is not None else []
        }


@dataclass
class Document:
    """文档
    """
    id: str
    name: str
    bookmarks: list[BookMark]
    parser: Parser

    def set_knowledgepoints_by_llm(self,
                                   llm: LLM,
                                   prompt: Prompt,
                                   self_consistency=False,
                                   samples: int = 5,
                                   top: float = 0.5) -> None:
        """ 使用 LLM 抽取知识点存储到 BookMark 中

        Args:
            llm (LLM): 指定 LLM
            prompt (Prompt): 使用的提示词类
            self_consistency (bool, optional): 是否采用自我一致性策略 (需要更多的模型推理次数). Defaults to False.
            samples (int, optional): 采样自我一致性策略的采样次数. Defaults to 5.
            top (float, optional): 采样自我一致性策略时，出现次数超过 top * samples 时才会被采纳，范围为 [0, 1]. Defaults to 0.5.
        """

        entitys: list[KPEntity] = []  # 复用知识点实体

        def set_knowledgepoints(bookmarks: list[BookMark]) -> None:
            for bookmark in bookmarks:
                if bookmark.title in ignore_page:
                    continue
                if bookmark.subs and isinstance(bookmark.subs[-1], BookMark):
                    set_knowledgepoints(bookmark.subs)
                else:
                    logger.success('子章节: ' + bookmark.title)
                    # 获取书签对应的页面内容
                    contents: list[Content] = []
                    for pg in range(bookmark.page_index,
                                    bookmark.page_end + 1):

                        # 起始页内容定位
                        page_contents = self.parser.get_page(pg).contents
                        if pg == bookmark.page_index:
                            idx = 0
                            for i, content in enumerate(page_contents):
                                blank_pattern = re.compile(
                                    r'\s+')  # 可能会包含一些空白字符这里去掉
                                if content.type == ContentType.Title and re.sub(
                                        blank_pattern, '',
                                        content.content) == re.sub(
                                            blank_pattern, '', bookmark.title):
                                    idx = i + 1
                                    break
                            page_contents = page_contents[idx:]
                        # 终止页内容定位
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
                    # 防止生成全空白
                    text_contents = text_contents.strip()
                    if len(text_contents) == 0:
                        bookmark.subs = []
                        continue
                    generate: list[list[
                        str]] = get_knowledgepoints(  # [head: str, relation: str, tail: str]
                            text_contents,
                            self_consistency=self_consistency,
                            samples=samples,
                            top=top)
                    logger.success('生成知识点三元组: ' + str(generate))
                    subs: list[KPTriple] = []
                    for triple in generate:
                        head, tail = None, None
                        for entity in entitys:
                            if triple[0] == entity.name:  # 需要实体归一化
                                head = entity
                                break
                        else:
                            head = KPEntity(id='2:' + str(uuid.uuid4()),
                                            name=triple[0])
                            entitys.append(head)
                        for entity in entitys:
                            if triple[2] == entity.name:
                                tail = entity
                                break
                        else:
                            tail = KPEntity(id='2:' + str(uuid.uuid4()),
                                            name=triple[2])
                            entitys.append(tail)
                        subs.append(
                            KPTriple(head=head,
                                     tail=tail,
                                     relation=KPRelation(id='4:' +
                                                         str(uuid.uuid4()),
                                                         type_name=triple[1])))
                    bookmark.subs = subs

        def get_knowledgepoints(content: str,
                                self_consistency=False,
                                samples: int = 5,
                                top: float = 0.5) -> list[str]:
            """ 使用 llm 生成知识点三元组列表

            Args:
                content (str): 输入文本
                self_consistency (bool, optional): 是否采用自我一致性策略 (需要更多的模型推理次数). Defaults to False.
                samples (int, optional): 采用自我一致性策略的采样次数. Defaults to 5.
                top (float, optional): 采样自我一致性策略时，出现次数超过 top * samples 时才会被采纳. Defaults to 0.5.

            Returns:
                list[str]: 生成的知识点三元组列表
            """
            if not self_consistency:
                # 默认策略：生成数量过多则重试，否则仍然过多则选择长度最长前5个
                retry = 0
                while True:
                    resp = llm.chat(prompt.get_prompt(content))
                    generate: list[list[str]] = prompt.post_process(resp)
                    if len(generate) <= 8 or retry >= 3:
                        break
                    retry += 1
                if len(generate) > 10:
                    # 随机选择5个
                    generate = random.sample(generate, 5)
                return generate
            else:
                # 自我一致性策略
                all_generate = []
                for idx in range(samples):
                    # 进行采样
                    resp = llm.chat(prompt.get_prompt(content))
                    logger.info(f'第{idx}次采样: ' + resp)
                    generate = prompt.post_process(resp)
                    logger.info(f'获取知识点三元组: ' + str(generate))
                    all_generate.append(generate)
                # 选择出现次数超过半数的进行返回 (SC分数过滤)
                generate = Counter(
                    [item for sublist in all_generate for item in sublist])
                return [
                    point for point, count in generate.items()
                    if count > (samples * top)
                ]

        set_knowledgepoints(self.bookmarks)

    def get_cyphers(self) -> list[str]:
        """ 将整体的关联关系存入图数据库中

        Returns:
            list[str]: 多条 cypher 语句
        """

        created_entity_ids = []
        cypher = f'CREATE (:Document {{id: "{self.id}", name: "{self.name}"}})'

        def bookmarks_to_cypher(bookmarks: list[BookMark], parent_id: str):
            cyphers: list[str] = []
            for bookmark in bookmarks:
                if bookmark.title in ignore_page:
                    continue
                cyphers.append(
                    f'CREATE (:Chapter {{id: "{bookmark.id}", name: "{bookmark.title}", page_start: {bookmark.page_index}, page_end: {bookmark.page_end}}})'
                )
                cyphers.append(  # 这里写不写类别无所谓
                    f'MATCH (n1 {{id: "{parent_id}"}}) MATCH (n2:Chapter {{id: "{bookmark.id}"}}) CREATE (n1)-[:子章节]->(n2)'
                )
                if bookmark.subs and isinstance(bookmark.subs[-1], BookMark):
                    cyphers.extend(
                        bookmarks_to_cypher(bookmark.subs, bookmark.id))
                elif bookmark.subs and isinstance(bookmark.subs[-1], KPTriple):
                    entitys: list[KPEntity] = []  # 复用知识点实体
                    for triple in bookmark.subs:
                        if triple.head not in entitys:
                            entitys.append(triple.head)
                        if triple.tail not in entitys:
                            entitys.append(triple.tail)
                    for entity in entitys:
                        if entity.id in created_entity_ids:
                            cyphers.append(
                                f'MATCH (n1:Chapter {{id: "{bookmark.id}"}}) MATCH (n2:KnowledgePoint {{id: "{entity.id}"}}) CREATE (n1)-[:提到知识点]->(n2)'
                            )
                        else:
                            cyphers.append(
                                f'MATCH (n1:Chapter {{id: "{bookmark.id}"}}) CREATE (n2:KnowledgePoint {{id: "{entity.id}", name: "{entity.name}"}}) CREATE (n1)-[:提到知识点]->(n2)'
                            )
                            created_entity_ids.append(entity.id)
                    for triple in bookmark.subs:
                        cyphers.append(
                            f'MATCH (n1:KnowledgePoint {{id: "{triple.head.id}"}}) MATCH (n2:KnowledgePoint {{id: "{triple.tail.id}"}}) CREATE (n1)-[:{triple.relation.type_name} {{id: "{triple.relation.id}"}}]->(n2)'
                        )
            return cyphers

        cyphers = [cypher]
        cyphers.extend(bookmarks_to_cypher(self.bookmarks, self.id))
        return cyphers
