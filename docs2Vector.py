import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone
from secret_key import open_ai_key, pinecone_api_key, pinecone_env

os.environ["OPENAI_API_KEY"] = open_ai_key
directory = 'C:\\Users\\HP\\ML Models\\langchain\\chatlawyer\\data'
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

documents = load_docs(directory)

def split_docs(documents, chunk_size=512, chunk_overlap=24):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

# query = embeddings.embed_query("Hello World!")

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

index_name = 'chat-lawyer'
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)