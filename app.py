from flask import Flask, render_template, jsonify, request
from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os




app = Flask(__name__)

load_dotenv()

os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "tester-ai-bot"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROK_API_KEY"] = GROK_API_KEY  

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROK_API_KEY"] = GROK_API_KEY  


embeddings = download_embeddings()

index_name = "tester-ai-bot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
api_key = os.getenv("GROK_API_KEY")
if api_key:
    api_key = api_key.strip() 
chatModel = ChatGroq(
    model="qwen/qwen3-32b",
    groq_api_key=api_key
    )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


import re

def parse_reasoning_model_output(text):
    # This regex removes everything between <think> and </think> tags
    # including the tags themselves.
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also clean up any trailing "Enough thinking" markers if they exist
    cleaned_text = cleaned_text.replace("Enough thinking", "")
    return cleaned_text.strip()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    # 1. Get the raw response from the RAG chain
    response = rag_chain.invoke({"input": msg})
    raw_answer = response["answer"]
    
    # 2. Clean the answer to remove the internal monologue
    clean_answer = parse_reasoning_model_output(raw_answer)
    
    print("Response : ", clean_answer)
    return str(clean_answer)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)