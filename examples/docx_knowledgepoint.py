import os
import sys

sys.path.append(os.getcwd())

from courseknow.document_parser import DOCXParser
from courseknow.database import Neo4j
from courseknow.llm import CoTPrompt, VLLM

model = VLLM('model/Qwen/Qwen2-7B-Instruct', stop_token_ids=[151329, 151336, 151338])
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
files = []

for file in files:
    with DOCXParser(file) as parser:
        document = parser.get_document()
        document.set_knowledgepoints_by_llm(model,
                                            CoTPrompt(),
                                            self_consistency=True,
                                            samples=6,
                                            top=0.8)
        neo.run(document.get_cyphers())
