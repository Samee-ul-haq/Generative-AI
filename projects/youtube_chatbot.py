from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


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
# question="Did they talk about aliens,if yes what?"
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

# results = retriever.invoke(question)

# print(results)



prompt=PromptTemplate(
    template="""
    You are a very helpful assistant.
    Answer from only the provided transcript context.
    If the context is insuffucient,just say you don't know.
    {context} 
    Question: {question}
    """,
    input_varaibles = ['context','question']
)



llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)




# context_text = "\n\n".join(doc.page_content for doc in results)
# final_prompt = prompt.invoke({"context":context_text,"question":question})

# answer = model.invoke(final_prompt)
# print(answer.content)





""" USING CHAINS """

from langchain_core.runnables import RunnableParallel , RunnablePassthrough , RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def format_document(results):
    context_text = "\n\n".join(doc.page_content for doc in results)
    return context_text


parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_document),
    "question" : RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser
answer = main_chain.invoke("Can you summarize the vedio")
print(answer)