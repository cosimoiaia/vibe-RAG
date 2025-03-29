import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langgraph.graph import StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import Dict, List, Tuple, TypedDict, Annotated
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

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
)

# Define state types
class AgentState(TypedDict):
    messages: List[Dict]
    documents: List[Dict]
    current_step: str
    query: str
    answer: str

# Create prompt template
prompt = PromptTemplate.from_template("""
Answer the following question based on the provided context:

Context: {context}
Question: {input}

Answer:""")

# Define node functions
def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant documents based on the query."""
    docs = vectorstore.similarity_search(
        state["query"],
        k=int(os.getenv("RETRIEVER_K", "4"))
    )
    state["documents"] = docs
    state["current_step"] = "generate"
    return state

def generate(state: AgentState) -> AgentState:
    """Generate answer using the retrieved documents."""
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

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Add edges
workflow.add_edge("retrieve", "generate")

# Set entry point
workflow.set_entry_point("retrieve")

# Set end point
workflow.set_finish_point("generate")

# Compile the graph
chain = workflow.compile()

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
        
        # Use the Qdrant wrapper to add documents
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
    # Initialize state
    state = {
        "messages": [],
        "documents": [],
        "current_step": "retrieve",
        "query": query,
        "answer": ""
    }
    
    # Run the graph
    result = chain.invoke(state)
    
    # Add to history
    st.session_state.history.append((query, result["answer"], result["documents"]))
    
for query, answer, sources in st.session_state.history:
    st.markdown(f"**Question:** {query}")
    st.markdown(f"**Answer:** {answer}")
    st.markdown(f"**Sources:** {', '.join([doc.metadata['source'] for doc in sources])}")

