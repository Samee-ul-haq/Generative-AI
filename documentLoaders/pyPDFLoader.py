from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('something.pdf')

docs=loader.load() # call the loader function that will load the pdf into the varaibles provided.

