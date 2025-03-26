# Vibe-RAG

A powerful Retrieval-Augmented Generation (RAG) application that enables users to interact with their PDF documents through natural language queries. Built with modern AI technologies and a user-friendly interface.

## Overview

Vibe-RAG combines the power of large language models with efficient document retrieval to provide accurate, context-aware answers to questions about your PDF documents. The application features a clean, intuitive interface and provides source attribution for all generated answers.

### Key Features
- 📄 **Drag-and-Drop PDF Upload:** Seamlessly upload and process PDF documents
- 💬 **Interactive Chat Interface:** Ask questions and receive context-aware answers
- 🔍 **Source Attribution:** View the specific document sections used to generate answers
- 🚀 **Fast Response Times:** Powered by Groq's high-performance LLM API
- 🔒 **Secure Document Processing:** Your documents are processed securely and stored in Pinecone's vector database

## Technology Stack

- **Frontend:** Streamlit
- **Vector Database:** Pinecone
- **Language Model:** Groq
- **Document Processing:** LangChain
- **PDF Processing:** PyPDF2

## Prerequisites

- Python 3.7 or higher
- API keys for:
  - Pinecone
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
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_ENVIRONMENT=your-pinecone-env
   GROQ_API_KEY=your-groq-api-key
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
└── README.md          # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Vector storage powered by [Pinecone](https://www.pinecone.io/)
- LLM API provided by [Groq](https://groq.com/)
- UI framework by [Streamlit](https://streamlit.io/)

