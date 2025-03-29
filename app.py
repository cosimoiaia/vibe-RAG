"""
Vibe-RAG: A Retrieval-Augmented Generation application for PDF document interaction.

This application allows users to upload PDF documents and ask questions about their content.
It uses Qdrant Cloud for vector storage, LangGraph for workflow management, and Groq for LLM inference.
"""

import streamlit as st
from typing import Dict, List, Tuple, TypedDict, Annotated
import io
import os
from dotenv import load_dotenv

# Third-party imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langgraph.graph import StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Type definitions
class AgentState(TypedDict):
    """Type definition for the state maintained throughout the LangGraph workflow.
    
    Attributes:
        messages: List of conversation messages
        documents: List of retrieved documents
        current_step: Current step in the workflow
        query: User's question
        answer: Generated answer
    """
    messages: List[Dict]
    documents: List[Dict]
    current_step: str
    query: str
    answer: str

# Initialize vector store and embeddings
def initialize_vector_store() -> Qdrant:
    """Initialize and configure the Qdrant vector store.
    
    Returns:
        Qdrant: Configured vector store instance
    """
    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", None)
    )

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Initialize or create collection
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

    # Initialize and return vectorstore
    return Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )

# Initialize LLM
def initialize_llm() -> ChatGroq:
    """Initialize the Groq LLM with configuration from environment variables.
    
    Returns:
        ChatGroq: Configured LLM instance
    """
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

# Define LangGraph workflow nodes
def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant documents based on the user's query.
    
    Args:
        state: Current state containing the query
        
    Returns:
        AgentState: Updated state with retrieved documents
    """
    docs = vectorstore.similarity_search(
        state["query"],
        k=int(os.getenv("RETRIEVER_K", "4"))
    )
    state["documents"] = docs
    state["current_step"] = "generate"
    return state

def generate(state: AgentState) -> AgentState:
    """Generate an answer using the retrieved documents and LLM.
    
    Args:
        state: Current state containing documents and query
        
    Returns:
        AgentState: Updated state with generated answer
    """
    context = "\n".join([doc.page_content for doc in state["documents"]])
    response = llm.invoke(
        prompt.format(
            context=context,
            input=state["query"]
        )
    )
    state["answer"] = response.content
    state["current_step"] = "end"
    return state

# Initialize components
vectorstore = initialize_vector_store()
llm = initialize_llm()

# Define prompt template
prompt = PromptTemplate.from_template("""
Answer the following question based on the provided context:

Context: {context}
Question: {input}

Answer:""")

# Create and configure the LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Configure workflow
workflow.add_edge("retrieve", "generate")
workflow.set_entry_point("retrieve")
workflow.set_finish_point("generate")

# Compile the workflow
chain = workflow.compile()

# File handling functions
def handle_file_upload(file) -> None:
    """Process and store an uploaded PDF file in the vector store.
    
    Args:
        file: Streamlit UploadedFile object containing the PDF
    """
    temp_filename = f"temp_{file.name}"
    
    try:
        # Save the uploaded file temporarily
        with open(temp_filename, "wb") as f:
            f.write(file.getvalue())
        
        # Load and process the PDF
        loader = PyPDFLoader(temp_filename)
        documents = loader.load_and_split()
        
        # Store documents in vector store
        vectorstore.add_documents(documents)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Streamlit UI
def main():
    """Main Streamlit application interface."""
    st.title("Vibe-RAG")

    # Initialize session staten
    if 'history' not in st.session_state:
        st.session_state.history = []

    # File upload interface
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    for file in uploaded_files:
        handle_file_upload(file)

    # Chat interface
    query = st.text_input("Ask a question:")
    if st.button("Submit"):
        # Initialize state for the workflow
        state = {
            "messages": [],
            "documents": [],
            "current_step": "retrieve",
            "query": query,
            "answer": ""
        }
        
        # Run the workflow
        result = chain.invoke(state)
        
        # Update conversation history
        st.session_state.history.append((query, result["answer"], result["documents"]))
    
    # Display conversation history
    for query, answer, sources in st.session_state.history:
        st.markdown(f"**Question:** {query}")
        st.markdown(f"**Answer:** {answer}")
        st.markdown(f"**Sources:** {', '.join([doc.metadata['source'] for doc in sources])}")

if __name__ == "__main__":
    main()

