import streamlit as st
import tempfile
import os
from ingest import load_document, split_document, embed_and_store
from query_langchain import build_qa_chain

st.title("PhD Research Assistant")
st.write("Upload your documents and ask questions.")

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_files = st.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]

    if new_files:
        for uploaded_file in new_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                docs = load_document(tmp_path)
                chunks = split_document(docs)
                embed_and_store(chunks, filename=uploaded_file.name)

                st.session_state.processed_files.append(uploaded_file.name)
                os.unlink(tmp_path)

        st.session_state.qa_chain = build_qa_chain()
        st.success(f"Processed {len(st.session_state.processed_files)} document(s). You can now ask questions")

if st.session_state.qa_chain is not None:
    st.divider()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask a question about your documents:")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            chain, retriever = st.session_state.qa_chain
            response = chain.invoke(question)
            source_docs = retriever.invoke(question)

            sources = set()
            for doc in source_docs:
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                sources.add(source)

            answer = response.content
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.write(answer)

            st.write("**Sources:**")
            for source in sources:
                st.write(f"- {os.path.basename(source)}")
