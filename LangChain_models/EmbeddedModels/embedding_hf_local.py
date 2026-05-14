from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=HuggingFaceEmbeddings(mode_name="sentence-transformers/all-MiniLM-L6-v2")

text="Srinagar is the city of kashmir" # change to document...

vector=embedding.aembed_query(text)
print(str(vector))