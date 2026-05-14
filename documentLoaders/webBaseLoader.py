from langchain_community.document_loaders import WebBaseLoader
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
    template="Tell me about the secipications of {object}",
    input_variables=['object']
)

url=''

loader=WebBaseLoader(url)
docs=loader.lazy_load()

parser=StrOutputParser()

chain=prompt | model | parser

print(chain.invoke({'object':'MacBook'}))

