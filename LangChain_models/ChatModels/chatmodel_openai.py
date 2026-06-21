from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model='gpt-4o-mini',temperature=0.9)

response=model.invoke("What is the capital of Turkey?")

print(response.content)