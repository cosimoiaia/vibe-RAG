# Vibe-RAG

A powerful Retrieval-Augmented Generation (RAG) application that enables users to interact with their PDF documents through natural language queries. Built with modern AI technologies and a user-friendly interface.

## Overview

Vibe-RAG combines the power of large language models with efficient document retrieval to provide accurate, context-aware answers to questions about your PDF documents. The application features a clean, intuitive interface and provides source attribution for all generated answers.

### Key Features
- 📄 **Drag-and-Drop PDF Upload:** Seamlessly upload and process PDF documents
- 💬 **Interactive Chat Interface:** Ask questions and receive context-aware answers
- 🔍 **Source Attribution:** View the specific document sections used to generate answers
- 🚀 **Fast Response Times:** Powered by Groq's high-performance LLM API
- 🔒 **Secure Document Processing:** Your documents are processed securely and stored in Qdrant Cloud
- 🤖 **Efficient Embeddings:** Uses HuggingFace's sentence-transformers for document embeddings
- 📊 **Graph-Based Workflow:** Built with LangGraph for flexible and maintainable processing pipeline

## Technology Stack

- **Frontend:** Streamlit
- **Vector Database:** Qdrant Cloud
- **Language Model:** Groq
- **Workflow Engine:** LangGraph
- **Document Processing:** LangChain Community
- **PDF Processing:** PyPDF2
- **Embeddings:** HuggingFace (sentence-transformers) with all-MiniLM-L6-v2 model

## Prerequisites

- Python 3.7 or higher
- API keys for:
  - Qdrant Cloud
  - Groq

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/cosimoiaia/vibe-rag.git
   cd vibe-rag
   ```

2. **Set Up Environment Variables:**
   Create a `.env` file in the project root with the following content:
   ```
   QDRANT_URL=your-qdrant-cloud-url-here
   QDRANT_API_KEY=your-qdrant-api-key-here
   GROQ_API_KEY=your-groq-api-key-here

   # Optional: Configure the number of documents to retrieve (default: 4)
   # RETRIEVER_K=4

   # Optional: Configure the model to use (default: llama-3.3-70b-versatile)
   # GROQ_MODEL=llama-3.3-70b-versatile
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Using the Application:**
   - Upload PDF documents using the drag-and-drop interface
   - Type your questions in the chat input
   - View answers with source citations
   - Track conversation history

## Project Structure

```
vibe-rag/
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (create this)
└── readme.md          # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for workflow management
- Vector storage powered by [Qdrant Cloud](https://cloud.qdrant.io/)
- LLM API provided by [Groq](https://groq.com/)
- UI framework by [Streamlit](https://streamlit.io/)
- Embeddings powered by [HuggingFace](https://huggingface.co/) and [sentence-transformers](https://www.sbert.net/)

