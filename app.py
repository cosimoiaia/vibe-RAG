import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import BM25Retriever
import io
from dotenv import load_dotenv
import os
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY", None)
)

# Initialize embeddings with HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Qdrant collection
collection_name = "vibe-rag"
try:
    qdrant_client.get_collection(collection_name)
except:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

# Initialize Qdrant vectorstore
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embeddings
)

# Initialize Groq LLM with the latest model
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
)

# Define the retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": int(os.getenv("RETRIEVER_K", "4"))}
)

# Create a prompt template
prompt = PromptTemplate.from_template("""
Answer the following question based on the provided context:

Context: {context}
Question: {input}

Answer:""")

# Define the document chain with the prompt
document_chain = create_stuff_documents_chain(
    llm,
    prompt=prompt
)

qa_chain = create_retrieval_chain(retriever, document_chain)

# Simple reranker based on similarity scores
def rerank_documents(docs, query, top_k=None):
    if top_k is None:
        top_k = int(os.getenv("RETRIEVER_K", "4"))
    scores = [(doc, doc.metadata.get('score', 0)) for doc in docs]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scores[:top_k]]

# Function to handle file uploads
def handle_file_upload(file):
    # Create a unique temporary filename using the original filename
    temp_filename = f"temp_{file.name}"
    
    try:
        # Save the uploaded file temporarily
        with open(temp_filename, "wb") as f:
            f.write(file.getvalue())
        
        # Load and process the PDF
        loader = PyPDFLoader(temp_filename)
        documents = loader.load_and_split()
        
        # Use the LangChain Qdrant wrapper to add documents
        vectorstore.add_documents(documents)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Streamlit app
st.title("Vibe-RAG")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Drag and drop file upload
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
for file in uploaded_files:
    handle_file_upload(file)

# Chat interface
query = st.text_input("Ask a question:")
if st.button("Submit"):
    docs = retriever.get_relevant_documents(query)
    reranked_docs = rerank_documents(docs, query)
    result = qa_chain.invoke({
        "input": query
    })
    st.session_state.history.append((query, result['answer'], docs))
    
for query, answer, sources in st.session_state.history:
    st.markdown(f"**Question:** {query}")
    st.markdown(f"**Answer:** {answer}")
    st.markdown(f"**Sources:** {', '.join([doc.metadata['source'] for doc in sources])}")

