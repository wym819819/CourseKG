import os
import sys

sys.path.append(os.getcwd())

from courseknow.document_parser import PDFParser
from courseknow.database import Neo4j
from courseknow.llm import QwenAPI, DeepKEPrompt

qwen = QwenAPI()
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
files = ['assets/《深度学习入门：基于Python的理论与实现》高清中文版.pdf', 'assets/机器学习周志华.pdf']

for file in files:
    with PDFParser(file) as parser:
        document = parser.get_document()
        document.set_knowledgepoints_by_llm(qwen, DeepKEPrompt())
        neo.run(document.get_cyphers())
