import os
import sys

sys.path.append(os.getcwd())

from courseknow.document_parser import PDFParser
from courseknow.database import Neo4j
from courseknow.llm import CoTPrompt, QwenAPI

qwen = QwenAPI(api_type='qwen-max-0403')
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
files = ['assets/机器学习.pdf']

for file in files:
    with PDFParser(file) as parser:
        document = parser.get_document()
        document.set_knowledgepoints_by_llm(qwen,
                                            CoTPrompt(example=False),
                                            self_consistency=False)
        neo.run(document.get_cyphers())
