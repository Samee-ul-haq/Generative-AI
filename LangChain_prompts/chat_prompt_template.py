from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

chatTemplate=ChatPromptTemplate([
    ('system',"You are a helpful {domain} expert"),
    ('human',"Explain me in simple terms,What is {topic}")
    ])

prompt=chatTemplate.invoke({'domain': 'Cricket','topic':'Bouncer'})

print(prompt)