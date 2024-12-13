# -*- coding: utf-8 -*-
"""Chat with PDFs

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1je5ihygZWftYwIqsC6GTA1BzMofGnefw
"""

# Install streamlit
!pip install streamlit

# Install dotenv
!pip install python-dotenv

# Install PyPDF2
!pip install PyPDF2

# Install langchain
!pip install langchain

# Install FAISS for vector storage
!pip install faiss-cpu

# Install HuggingFace Transformers
!pip install transformers

# Install htmlTemplates (if custom, ensure it's available in your project folder)
# Create a placeholder if necessary for your project

# Install Rust for tokenizers if needed
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
!source $HOME/.cargo/env

!pip install -U langchain-community

import streamlit
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("hkunlp/instructor-xl")
model = AutoModel.from_pretrained("hkunlp/instructor-xl")

import os
import streamlit as st

# Function to read text from multiple PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a FAISS vector store
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational chain using the LLM
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display chat
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"**User:** {message.content}")
        else:
            st.markdown(f"**Bot:** {message.content}")

# Main function for Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚", layout="wide")
    st.title("Chat with Multiple PDFs 📚")

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload Your PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDF files here:",
            type=["pdf"],
            accept_multiple_files=True
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    # Step 1: Get text from PDFs
                    raw_text = get_pdf_text(pdf_docs)

                    # Step 2: Split text into chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Step 3: Create FAISS vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Step 4: Create conversational chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Documents processed successfully! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main area for chat interaction
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)
    elif user_question:
        st.warning("Please process your documents before asking questions.")

if __name__ == "__main__":
    main()

!pip install pyngrok

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# # Copy your Streamlit code here
# import os
# import streamlit as st
# 
# 
# # Define functions and Streamlit main function here
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text
# 
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks
# 
# def get_vectorstore(text_chunks):
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore
# 
# def get_conversation_chain(vectorstore):
#     llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain
# 
# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']
# 
#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.markdown(f"**User:** {message.content}")
#         else:
#             st.markdown(f"**Bot:** {message.content}")
# 
# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚", layout="wide")
#     st.title("Chat with Multiple PDFs 📚")
# 
#     with st.sidebar:
#         st.subheader("Upload Your PDFs")
#         pdf_docs = st.file_uploader(
#             "Upload your PDF files here:",
#             type=["pdf"],
#             accept_multiple_files=True
#         )
#         if st.button("Process"):
#             if pdf_docs:
#                 with st.spinner("Processing documents..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     vectorstore = get_vectorstore(text_chunks)
#                     st.session_state.conversation = get_conversation_chain(vectorstore)
#                     st.success("Documents processed successfully! You can now ask questions.")
#             else:
#                 st.warning("Please upload at least one PDF file.")
# 
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None
# 
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question and st.session_state.conversation:
#         handle_userinput(user_question)
#     elif user_question:
#         st.warning("Please process your documents before asking questions.")
# 
# if __name__ == "__main__":
#     main()
#

!streamlit run app.py & npx localtunnel --port 8501