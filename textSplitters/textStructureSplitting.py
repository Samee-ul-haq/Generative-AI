from langchain_text_splitters import RecursiveCharacterTextSplitter

text=""""""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks=splitter.split_text(text)
