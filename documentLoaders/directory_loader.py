from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)
# can use lazy_load which will give output without glitching at the very first..
docs=loader.load() 

print(docs[0].page_content)
print(docs[0].metadata)