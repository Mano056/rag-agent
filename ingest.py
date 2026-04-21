from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys


load_dotenv()

def load_document(path):
    if path.endswith(".txt"):
        loader = TextLoader(path)
    elif path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    return loader.load()

def split_document(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)

def embed_and_store(chunks, filename=None):
    if filename:
        for chunk in chunks:
            chunk.metadata["source"] = filename
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return db

if __name__ == "__main__":
    docs = load_document(sys.argv[1])
    chunks = split_document(docs)
    embed_and_store(chunks)