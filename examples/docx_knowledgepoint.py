from coursekg.document_parser import DOCXParser
from coursekg.database import Neo4j
from coursekg.llm import ExamplePrompt, VLLM, ExamplePromptStrategy

model = VLLM(path='model/Qwen/Qwen2-7B-Instruct')
neo = Neo4j('http://10.4.3.67:7474', 'neo4j', 'neo4j')
files = ['assets/探索数据的奥秘.docx']
strategy = ExamplePromptStrategy(
    embed_model_path='model/lier007/xiaobu-embedding-v2')

for file in files:
    with DOCXParser(file) as parser:
        document = parser.get_document()
        document.set_knowledgepoints_by_llm(model,
                                            ExamplePrompt(strategy),
                                            self_consistency=True,
                                            samples=6,
                                            top=0.8)
        neo.run(document.get_cyphers())
