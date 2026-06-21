from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Write a Joke about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Give the explanation of the following joke -- {text}",
    input_variables=['text']
)
parser=StrOutputParser()

joke_gen=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model,parser)
})

chain=RunnableSequence(joke_gen,parallel_chain)

print(chain.invoke({'topic':'Cricket'})) 