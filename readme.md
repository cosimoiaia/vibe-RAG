### Vibe-RAG

**Vibe-RAG** is a Retrieval-Augmented Generation (RAG) application built using LangChain, Pinecone for the vector database, Streamlit for the user interface, and Groq for the Language Model API. This application allows users to upload PDF documents, ask questions, and receive answers based on the content of the uploaded documents. The app also provides the sources used to generate the answers.

#### Features
- **Drag-and-Drop PDF Upload:** Easily upload PDF documents to the application.
- **Chat Interface:** Ask questions and receive answers based on the uploaded documents.
- **Source Attribution:** View the sources used to generate the answers.

#### Prerequisites
- Python 3.7 or higher
- API keys for Pinecone and Groq

#### Installation

1. **Clone the Repository (Optional):**
   If you have the files locally, you can skip this step. Otherwise, clone the repository:
   ```bash
   git clone https://github.com/cosimoiaia/vibe-rag.git
   cd vibe-rag
   ```

2. **Create a `.env` File:**
   Save the following content in a file named `.env`:

   ```
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_ENVIRONMENT=your-pinecone-env
   GROQ_API_KEY=your-groq-api-key
   ```

3. **Install Dependencies:**
   Open a terminal, navigate to the directory where `requirements.txt` and `.env` are located, and run the following command to install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

#### Running the Application

1. **Run the Streamlit App:**
   Open a terminal, navigate to the directory where `app.py`, `requirements.txt`, and `.env` are located, and run the following command:

   ```bash
   streamlit run app.py
   ```

2. **Interact with the App:**
   - **Upload PDFs:** Use the drag-and-drop interface to upload PDF documents.
   - **Ask Questions:** Enter your questions in the text input box and click "Submit" to get answers.
   - **View Sources:** The app will display the answers along with the sources used to generate the answers.

#### Notes
- **API Keys:** Ensure you have valid API keys for Pinecone and Groq and replace `your-pinecone-api-key`, `your-pinecone-env`, and `your-groq-api-key` in the `.env` file with your actual API keys.
- **Reranker:** The reranker is a simple implementation that sorts documents based on similarity scores. You can enhance it using more sophisticated techniques if needed.

#### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

