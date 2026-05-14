from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv

# 1:14:12

chat_templete=ChatPromptTemplate([
    ('system','You are a helpfull customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','query')
])

chat_history=[]

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

prompt=chat_templete.invoke({'chat_history':chat_history},{'query':'what is the status of my order'})

print(chat_history)