# -*- coding: utf-8 -*-
# Create Date: 2024/07/15
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: examples/docx_knowledgepoint.py
# Description: 为docx格式文档抽取知识点图谱

from coursekg.document_parser import DOCXParser
from coursekg.database import Neo4j
from coursekg.llm import ExamplePrompt, VLLM

model = VLLM(path='model/Qwen/Qwen2-7B-Instruct')
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
files = ['assets/探索数据的奥秘.docx']

for file in files:
    with DOCXParser(file) as parser:
        document = parser.get_document()
        document.set_knowledgepoints_by_llm(model,
                                            ExamplePrompt(),
                                            self_consistency=True,
                                            samples=6,
                                            top=0.8)
        neo.run(document.to_cyphers())
