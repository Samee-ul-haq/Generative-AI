from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader=PyPDFLoader('temp.pdf')

docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

text="""

"""
result=splitter.split_text(text)
# result=splitter.split_document(docs)

print(result)