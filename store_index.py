from src.helper import text_split, minimal_docs, embedding
from dotenv import load_dotenv
import os
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROK_API_KEY"] = GROK_API_KEY   
from pinecone import Pinecone 
pinecone_api_key = PINECONE_API_KEY

pc = Pinecone(api_key=pinecone_api_key)
from pinecone import ServerlessSpec

index_name = "tester-ai-bot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

texts_chunks = text_split(minimal_docs)

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embedding=embedding,    
    index_name=index_name
)
#add more data to existing index  
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
retrieved_docs = retriever.invoke("What is software testiing?")

from langchain_groq import ChatGroq
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    api_key = api_key.strip() 
chatModel = ChatGroq(
    model="qwen/qwen3-32b",
    groq_api_key=api_key
    )
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate