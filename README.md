# Chat with Multiple PDFs using Python & Gemini Pro 🐍📚

A Python application that enables dynamic conversations with multiple PDF documents using Google's Gemini Pro LLM and a Streamlit interface.

## 🌟 Features
- **Upload and process multiple PDF documents simultaneously**
- **Interactive chat interface for document queries**
- **Real-time document processing with status updates**
- **Vector store implementation using FAISS for efficient document retrieval**
- **Secure API key management through environment variables**
- **User-friendly Streamlit interface with clear feedback**

---

## 🛠️ Tech Stack
- **Core Language:** Python 3.x  
- **Framework:** Streamlit  
- **Language Model:** Google Gemini Pro  
- **Vector Store:** FAISS  
- **PDF Processing:** PyPDF2  
- **Embeddings:** Google Generative AI Embeddings  
- **Chain Management:** LangChain  

---

## 🔑 Key Components
1. **PDF Text Extraction**  
   - Efficiently extracts text from multiple PDFs using `PyPDF2`.

2. **Text Chunking**  
   - Implements recursive character text splitting for optimal processing.

3. **Vector Store**  
   - Utilizes `FAISS` for similarity search and document retrieval.

4. **Conversational Chain**  
   - Features a custom prompt template for accurate and detailed responses.

5. **Error Handling**  
   - Includes robust error management for PDF processing and API interactions.

---

## 📦 Dependencies
Ensure the following Python packages are installed to run the application:
- `streamlit`
- `google-generativeai`
- `python-dotenv`
- `langchain`
- `PyPDF2`
- `chromadb`
- `faiss-cpu`
- `langchain_google_genai`

---

**Contributor:** Kriti C Parikh

References: Krish Naik
