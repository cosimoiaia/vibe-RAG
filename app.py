import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.qa_with_sources import QASourceChain
from langchain.retrievers import BM25Retriever
from PyPDF2 import PdfReader
import io
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize Pinecone index
index_name = "vibe-rag"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)
index = pinecone.Index(index_name)

# Initialize Groq LLM
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define the retriever
retriever = Pinecone.from_existing_index(index_name, embeddings)

# Define the chain
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# Simple reranker based on similarity scores
def rerank_documents(docs, query, top_k=5):
    scores = [(doc, doc.metadata.get('score', 0)) for doc in docs]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scores[:top_k]]

# Function to handle file uploads
def handle_file_upload(file):
    loader = PyPDFLoader(file)
    documents = loader.load_and_split()
    texts = [doc.page_content for doc in documents]
    metadatas = [{"source": doc.metadata["source"]} for doc in documents]
    vectors = embeddings.embed_documents(texts)
    ids = [str(i) for i in range(len(texts))]
    index.upsert(vectors=vectors, ids=ids, metadata=metadatas)

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
    result = qa_chain({"input_documents": reranked_docs, "question": query})
    st.session_state.history.append((query, result['answer'], result['source_documents']))
    
for query, answer, sources in st.session_state.history:
    st.markdown(f"**Question:** {query}")
    st.markdown(f"**Answer:** {answer}")
    st.markdown(f"**Sources:** {', '.join([doc.metadata['source'] for doc in sources])}")
