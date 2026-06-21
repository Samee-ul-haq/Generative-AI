from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm1=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

# llm2=HuggingFaceEndpoint(
#     repo_id=""
# )

model= ChatHuggingFace(llm=llm1)

prompt1=PromptTemplate(
    template="Generate short and simplr notes from the {text}",
    input=['text']
)

prompt2=PromptTemplate(
    template="Generate 5 questions answers from the following text \n {text}",
    input_varaible=['text']
)

prompt3=PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes}",
    input_varaible=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz' : prompt2 | model | parser
})

merge_chain=prompt3 | model | parser

chain=parallel_chain | merge_chain

text="""Machine Learning (ML) is a branch of Computer Science and Artificial Intelligence that enables computers to learn from data 
and improve their performance without being explicitly programmed for every task.
 Instead of following fixed instructions, machine learning systems identify patterns, make predictions, and adapt based on experience.
Machine learning is widely used in everyday life. Examples include recommendation systems on streaming platforms, voice assistants,
 spam email filters, facial recognition, online shopping suggestions, and self-driving vehicles. Companies and researchers use machine 
 learning to analyze large amounts of data quickly and accurately."""

response=chain.invoke({'text':text})

print(response)