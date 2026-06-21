from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='Write a summary for the following poen --- \n {poem}',
    input_variables={'poem'}
)

parser=StrOutputParser()

loader=TextLoader('Cricket.txt',encoding='utf-8')

docs=loader.load()

chain=prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))

print(type.docs())
print(docs[0].page_content)

print(docs[0].metadata)
