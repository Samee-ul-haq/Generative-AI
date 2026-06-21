from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings


url='https://youtu.be/Gfr50f6ZBvo?si=jrmFbfrL35zV5Wdl'
loader=YoutubeLoader.from_youtube_url(url,add_video_info=False)


docs=loader.load()
print(type(docs))



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)    
print(len(texts))



# One way of creating embeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="intfloat/e5-large-v2",
#     encode_kwargs={"prompt": "passage: "},
#     query_encode_kwargs={"prompt": "query: "},
# )



#Another way of creating embeddings
query="Give me the summary of the vedio"
underlying_embeddings = HuggingFaceEmbeddings()



store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model_name
)


vectorstore = InMemoryVectorStore(cached_embedder)
vectorstore.add_documents(texts)



retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)

retriever.invoke(query)

# results = vectorstore.similarity_search(query)
# vector = cached_embedder.embed_query(query)