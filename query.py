from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

def search(query, db):
    results = db.similarity_search(query, k=3)
    return results

def answer(query, db):
    chunks = search(query, db)
    context = "\n\n".join(chunks if isinstance(chunks[0], str) else [chunk.page_content for chunk in chunks])

    prompt = "Answer the question based on the context below. \n\nContext:\n" + context + "\n\nQuestion: " + query + "\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    db = load_db()
    while True:
        query = input("\nAsk a question (or 'quit): ")
        if query.lower() == "quit":
            break
        response = answer(query, db)
        print(f"\nAnswer: {response}")