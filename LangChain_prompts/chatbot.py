from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model= ChatHuggingFace(llm=llm)

messages=[
    SystemMessage(content="You are a very helpful assistant"),
    HumanMessage(content="Tell me about the LangChain")
]

response=model.invoke(messages)

messages.append(AIMessage(content=response.content))

print(messages)