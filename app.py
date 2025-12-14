import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="🤖 PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("🤖 PDF Chatbot")
st.caption("Chat with your PDF documents using LLMs and RAG")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Process PDF
if uploaded_file:
    with st.spinner("📚 Processing PDF..."):
        pdf_path = "uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(docs, embeddings)

        llm = ChatGroq(
            model="llama-3.1-8b-instant"
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )

        st.session_state.qa_chain = qa_chain
        st.success("✅ PDF processed successfully! You can start chatting.")

# Chat UI
if st.session_state.qa_chain:
    query = st.chat_input("💬 Ask a question about your PDF")

    if query:
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.qa_chain.invoke(
                {"question": query}
            )
            answer = result["answer"]

            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("assistant", answer))

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.info("👆 Upload a PDF to begin")
