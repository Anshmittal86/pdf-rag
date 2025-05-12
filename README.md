# 📄 PDF RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to interact with the contents of a PDF through natural language queries. This project combines document processing, vector similarity search, and large language models to deliver a contextual question-answering experience.

## 🚀 Features

- Upload any PDF document
- Extracts and chunks text from PDF
- Generates vector embeddings using OpenAI
- Stores and searches chunks using FAISS (vector database)
- Injects context into prompts for LLM-based answers
- User-friendly chat interface (built with Streamlit)

## 🧠 Tech Stack

- **LangChain** – RAG pipeline and LLM integration
- **OpenAI GPT-3.5/GPT-4** – Answer generation
- **FAISS** – Vector similarity search
- **Streamlit** – Frontend interface
- **PyPDF2 / pdfplumber** – PDF text extraction
- **tiktoken** – Token management for chunking

## 📂 Folder Structure

pdf-rag/
├── app.py # Streamlit app
├── requirements.txt # Python dependencies
└── README.md # Project overview

## 🔧 Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Anshmittal86/pdf-rag.git
cd pdf-rag
```
2. Create and activate virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Create and activate virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Add your GEMINI API key
Create a .env file and add:
GEMINI_API_KEY=your_openai_api_key_here

5. Run the app
```bash
streamlit run app.py
```

## Example Use Cases
- Legal or technical document understanding
- Academic paper Q&A
- Internal documentation search assistant

## Future Enhancements
- Chat history and memory
- Support for DOCX, TXT, and web pages
