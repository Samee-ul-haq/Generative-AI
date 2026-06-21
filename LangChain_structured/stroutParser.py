#output parsers convert raw LLM resposes into structured formats like json,csv,
# can be used for both types of parsers.
# 4 Types of parsers
# 1 -- strOutputParser  : reposes can be direclty printed as string
# 2--  jsonOutputPaser  : 


from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

#1st prompt -> detailed report
template1=PromptTemplate(
    template='write a detailed report on {topic}',
    input_variable=['text']
)

#2nd prompt -> summary
template2=PromptTemplate(
    template='Write a 5 line sumaary on the follwing text ./n {text}',
    input_variable=['text']
)

parser=StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response=chain.invoke({'topic':'blackhole'})

print(response)