# RAG Agent

## What it does?
This is a small RAG agent that first breaks a selected text/PDF file into chunks using ingest.py. Then run the query.py to ask questions about the uploaded file, and get a response from the agent.

## Installation 
`git clone https://github.com/Mano056/rag-agent.git`
`cd rag-agent`
`pip install python-dotenv chromadb langchain langchain-community langchain-text-splitters groq sentence-transformers`

## Setup
Create a `.env` file in the project folder: 
`GROQ_API_KEY=your_groq_api_key_here`
Get your free API key at https://console.groq.com

## Usage
`python ingest.py yourfile.txt` or `python ingest.py yourfile.pdf`
`python query.py`

## Live app
[Click here for the live app]
(https://researchassist.streamlit.app/)
Follow link for a live PhD assistant. You could upload PDF/TXT files and ask questions related to the documents, as well as receive the source of the answers.