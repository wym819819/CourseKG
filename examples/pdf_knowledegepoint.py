from coursekg.document_parser import PDFParser
from coursekg.database import Neo4j
from coursekg.llm import CoTPrompt, VLLM

model = VLLM('model/Qwen/Qwen2-7B-Instruct')
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
files = ['assets/深度学习入门：基于Python的理论与实现.pdf', 'assets/机器学习.pdf']

for file in files:
    with PDFParser(file) as parser:
        document = parser.get_document()
        document.set_knowledgepoints_by_llm(model,
                                            CoTPrompt(),
                                            self_consistency=True,
                                            samples=6,
                                            top=0.8)
        neo.run(document.get_cyphers())
