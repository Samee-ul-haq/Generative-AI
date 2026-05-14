#in pydantic parsers we can inforce schema as well as validation and seamless integration

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

# problem with json implementaion is that we cant inforce schema...
# For that we can use structured output parser from predefined schema 
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name:str=Field(description="Name of a person")
    age:int =Field(description="Age of a person")
    city:str=Field(description="Where person belongs to")

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template="Generate the name,age,city of person from {place} \n {format_instruction}",
    input_variables=['place'],
    partial_varaibles={'format_instruction':parser.get_format_instruction()}
)

chain=template | model | parser

response=chain.invoke({'place':'Iran'})
print(response)
