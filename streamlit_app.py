import streamlit as st
from pathlib import Path
import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from concurrent.futures import ThreadPoolExecutor 
from itertools import chain #Flatten
import ast #Parsing

qdrant_url_link = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
collection_name = st.secrets["QDRANT_COLLECTION_NAME"]
google_api_key = st.secrets["GEMINI_API_KEY"]

# Set page configuration
st.set_page_config(
    page_title="PDF RAG AI Assistant",
    page_icon="📚",
    layout="wide",
)

# Add custom CSS
st.markdown(
    """
<style>
    .main-header {
    font-size: 2.5rem;
    color: #8ab4f8; /* Lighter blue for better visibility on dark */
    margin-bottom: 1rem;
    text-align: center;
}

.sub-header {
    font-size: 1.5rem;
    color: #9aa0a6; /* Muted gray, standard for dark UIs */
    margin-bottom: 2rem;
    text-align: center;
}

.stApp {
    background-color: #121212; /* Very dark gray, not pure black */
}

.chat-container {
    border-radius: 10px;
    padding: 20px;
    background-color: #1e1e1e; /* Slightly lighter than background */
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
}

.user-message {
    background-color: #1a73e8; /* Google Blue with enough contrast */
    color: #e8f0fe;
    padding: 10px 15px;
    border-radius: 15px 15px 0 15px;
    margin: 5px 0;
    max-width: 80%;
    align-self: flex-end;
}

.assistant-message {
    background-color: #2c2c2e; /* Dark gray for contrast */
    color: #f1f3f4;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0;
    margin: 5px 0;
    max-width: 80%;
    align-self: flex-start;
}

@media (prefers-color-scheme: light) {
    .main-header {
        color: #222;
    }
    .stMainBlockContainer.block-container.st-emotion-cache-zy6yx3.en45cdb4 {
        background-color: #f4f4f4;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "api_key" not in st.session_state:
    st.session_state.api_key = google_api_key
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

# Main layout
st.markdown('<h1 class="main-header">PDF RAG AI Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload PDFs, ask questions, and get AI-powered answers</p>',
    unsafe_allow_html=True,
)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Embedding and model settings
    st.subheader("Model Settings")
    embedding_model = st.selectbox("Embedding Model", ["models/text-embedding-004"])

    generation_model = st.selectbox(
        "Generation Model", ["gemini-2.0-flash-001", "gemini-1.5-pro-latest"]
    )

    # Advanced settings
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 100, 4000, 2000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 1000, 200)

    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.awaiting_response = False
        st.rerun()

# File upload section
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

# Process the uploaded PDF
if uploaded_file is not None and st.session_state.api_key:
    if uploaded_file.name not in st.session_state.processed_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name

            try:
                # Load the document
                loader = PyPDFLoader(file_path=pdf_path)
                docs = loader.load()

                # Split the document into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                split_docs = text_splitter.split_documents(documents=docs)

                # Initialize Google Generative AI Embeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=embedding_model, google_api_key=st.session_state.api_key
                )

                # Create or append to vector store
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = QdrantVectorStore.from_documents(
                        documents=split_docs,
                        url=qdrant_url_link,
                        api_key=qdrant_api_key,
                        collection_name=collection_name,
                        embedding=embeddings,
                    )
                else:
                    st.session_state.vector_store.add_documents(split_docs)

                # Setup retriever
                st.session_state.retriever = QdrantVectorStore.from_existing_collection(
                    url=qdrant_url_link,
                    api_key=qdrant_api_key,
                    collection_name=collection_name,
                    embedding=embeddings,
                )

                # Mark file as processed
                st.session_state.processed_files.append(uploaded_file.name)
                st.success(f"Successfully processed {uploaded_file.name}")

                # Clean up the temporary file
                os.unlink(pdf_path)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        st.info(f"{uploaded_file.name} has already been processed.")
elif uploaded_file is not None and not st.session_state.api_key:
    st.warning("Please enter your Google AI API Key in the sidebar first.")

# Display processed files
if st.session_state.processed_files:
    st.subheader("Processed Documents")
    for file in st.session_state.processed_files:
        st.write(f"- {file}")

# Horizontal line to separate sections
st.markdown("---")

# Chat interface
st.subheader("Ask Questions About Your Documents")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f"<div class='user-message'><strong>You:</strong> {message['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='assistant-message'><strong>Assistant:</strong> {message['content']}</div>",
                unsafe_allow_html=True,
            )


# Submit function for the form
def submit_query():
    if st.session_state.user_query:
        # Add user message to chat history
        st.session_state.messages.append(
            {"role": "user", "content": st.session_state.user_query}
        )
        # Set flag to process the query
        st.session_state.awaiting_response = True
        # Clear the input field after submission
        st.session_state.user_query = ""


# Query input form
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_input("Enter your question:", key="user_query")
    submit_button = st.form_submit_button("Submit", on_click=submit_query)

# Process query when needed
if (
    st.session_state.awaiting_response
    and st.session_state.retriever is not None
    and st.session_state.api_key
):
    # Get the last user message
    user_query = st.session_state.messages[-1]["content"]

    with st.spinner("Searching for relevant information..."):
        try:
            
            # Initialize Google Generative AI client
            client = genai.Client(api_key=st.session_state.api_key)
            
            system_prompt_for_subqueries = """
            You are a helpful AI Assistant. 
            Your task is to take the user query and break it down into different sub-queries.

            Rule:
            Minimum Sub Query Length :- 3
            Maximum Sub Query Length :- 5

            Example:
            Query: How to become GenAI Developer?
            Output: [
                "How to become GenAI Developer?",
                "What is GenAI?",
                "What is Developer?",
                "What is GenAI Developer?",
                "Steps to become GenAI Developer."
            ]
            """
            
            # Call Gemini API to break down the user's query into sub-queries
            breakdown_response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=f"Query: {user_query}",
                config=types.GenerateContentConfig(system_instruction=system_prompt_for_subqueries)
            )
            
            # Convert the Gemini response to a Python list (parse the output safely)
            sub_queries = ast.literal_eval(breakdown_response.text.strip())
            print("Sub Queries:", sub_queries)

            # === Reciprocal Rank Fusion ===

            retriever = st.session_state.retriever  # capture outside thread
            
            # Function to retrieve relevant document chunks for each sub-query
            def retrieve_chunks(query):
                return retriever.similarity_search(query=query)
            
            
            # Use ThreadPoolExecutor to perform parallel retrieval of chunks for each sub-query
            with ThreadPoolExecutor() as executor:
                all_chunks = list(executor.map(retrieve_chunks, sub_queries))

            # Helper to generate a unique ID for each chunk (or you can use doc.metadata['id'] if available)
            def get_doc_id(doc):
                return doc.page_content.strip()[:50]  # Use first 50 characters as an ID
            
            # Create rankings (lists of doc_ids per sub-query result)
            rankings = []
            for result in all_chunks:
                rankings.append([get_doc_id(doc) for doc in result])
                
            # Reciprocal Rank Fusion
            def reciprocal_rank_fusion(rankings, k=60):
                scores = {}
                for ranking in rankings:
                    for rank, doc_id in enumerate(ranking):
                        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
                sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                return [doc_id for doc_id, _ in sorted_docs]
            
            # Get final ranked doc IDs
            final_doc_ids = reciprocal_rank_fusion(rankings)

            # Map doc IDs to actual chunks
            doc_map = {get_doc_id(doc): doc for doc in chain.from_iterable(all_chunks)}
            ranked_chunks = [doc_map[doc_id] for doc_id in final_doc_ids if doc_id in doc_map]

            
            # === GENERATION PART ===

            # Prepare the final system prompt with the top-ranked chunks
            final_system_prompt = f"""
            You are a helpful assistant who answers the user's query using the following pieces of context.

            Context:
            {[doc.page_content for doc in ranked_chunks]}
            """

            # Generate response
            response = client.models.generate_content(
                model=generation_model,
                contents=user_query,
                config=types.GenerateContentConfig(system_instruction=final_system_prompt),
            )

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response.text}
            )

            # Reset the flag
            st.session_state.awaiting_response = False

            # Force a rerun to display the new messages
            st.rerun()

        except Exception as e:
            st.error(f"Error processing your question: {str(e)}")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"I'm sorry, I encountered an error: {str(e)}",
                }
            )
            st.session_state.awaiting_response = False
            st.rerun()

elif st.session_state.awaiting_response and (
    not st.session_state.retriever or not st.session_state.api_key
):
    if not st.session_state.api_key:
        st.warning("Please enter your Google AI API Key in the sidebar.")
    elif not st.session_state.retriever:
        st.warning("Please upload and process at least one PDF document first.")
    st.session_state.awaiting_response = False

# Footer
st.markdown("---")
st.markdown("PDF RAG AI Assistant - Powered by Google Generative AI and Streamlit")
