from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

documents=[
    "Srinagar is the capital of J &K",
    "Tujar is the village of Sopore",
    "This is my laptop on my lap"
]
result=embedding.embed_documents(documents)

print(str(result))