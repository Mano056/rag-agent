from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()

def build_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context below.
    
    Context: {context}
    Question: {question}
                    
    Answer:"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain, retriever


if __name__ == "__main__":
    chain = build_qa_chain()
    while True:
        query = input("\nAsk a question (or 'quit'): ")
        if query.lower() == "quit":
            break
        response = chain.invoke(query)
        print(f"\nAnswer: {response.content}")
