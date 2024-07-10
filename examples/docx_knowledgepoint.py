import os
import sys

sys.path.append(os.getcwd())

from coursekg.document_parser import DOCXParser
from coursekg.database import Neo4j
from coursekg.llm import CoTPrompt, QwenAPI

model = QwenAPI(url='http://10.4.0.141:1120/chat')
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
files = ['assets/探索数据的奥秘.docx']

for file in files:
    with DOCXParser(file) as parser:
        document = parser.get_document()
        document.set_knowledgepoints_by_llm(model,
                                            CoTPrompt(),
                                            self_consistency=True,
                                            samples=6,
                                            top=0.8)
        neo.run(document.get_cyphers())