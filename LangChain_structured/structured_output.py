from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()


# no data validation in ouputParsers...solution Pydantic output parser
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)


schema=[
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic'),
]

parser=StructuredOutputParser.from_respose_schema(schema)

template=PromptTemplate(
    template='Give 3 three facts about {topic} \n {forma_instruction}',
    variable_input=['topic'],
    partial_variable=({'format_instruction' : parser.get_format_instruction()})
)


chain = template | model | parser
result=chain.invoke({'topic':'black hole'})
print(result)