from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (RunnableSequence,
    RunnableParallel,RunnableLambda,RunnablePassthrough)

load_dotenv()

def word_count(word):
    return len(word.split())

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="Write a Joke about {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()

joke_gen=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    'text':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})

final_chain=RunnableSequence(joke_gen,parallel_chain)

print(final_chain.invoke({'topic':'AI'}))