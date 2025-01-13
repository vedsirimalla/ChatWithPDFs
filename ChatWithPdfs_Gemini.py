import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Google API Key not found. Please set it in your .env file.")
    st.stop()

def get_pdf_text(pdf_docs):
    text = ""
    processed_count = 0
    processed_pdfs = []
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
            processed_count += 1
            processed_pdfs.append(pdf.name)
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
    return text, processed_count, processed_pdfs

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
    , return_only_outputs=True)
    print(response)
    return response["output_text"]

def main():

    st.set_page_config("ChatWithMultiplePDFs", page_icon="📚", layout="wide")
    
    def load_css():
        with open("assets/style.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Call load_css() before any other Streamlit commands
    load_css()
    
    st.header("Chat with Multiple PDFs💻📱")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("📁 Document Upload")
        pdf_docs = st.file_uploader("Upload your PDF Files and click on 'Process'", accept_multiple_files=True)
        process_button = st.button("Process PDFs")
        if st.button("Clear Chat"):
            st.session_state.messages = []

    if process_button and pdf_docs:
        with st.spinner("Processing..."):
            raw_text, processed_count, processed_pdfs = get_pdf_text(pdf_docs)
            if processed_count > 0:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success(f"✅ Successfully processed {processed_count} PDF{'s' if processed_count > 1 else ''}")
                st.write("Processed PDFs:")
                for pdf_name in processed_pdfs:
                    st.write(f"- {pdf_name}")
            else:
                st.error("No PDFs were successfully processed. Please check your files and try again.")

    st.subheader("Ask a question about your PDFs")
    user_question = st.text_input("Enter your question here")

    if user_question:
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process PDFs before asking questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_question})
            response = user_input(user_question)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    if user_question:
        user_input(user_question)
        user_question = ""  # Clear the input after processing

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()

#References: Krish Naik