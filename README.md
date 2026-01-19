# ask-my-pdf

Project overview
----------------
ask-my-pdf is a Streamlit-based Retrieval-Augmented Generation (RAG) demo that lets a user upload a PDF and ask questions about its content. The system extracts text from the PDF, creates embeddings, stores them in a FAISS index, retrieves relevant chunks for a user query, and uses an LLM (Mistral 7B via OpenRouter) to generate answers strictly from the retrieved document context.

How it works (high-level RAG flow)
----------------------------------
1. User uploads a PDF through the Streamlit UI.
2. PDF is loaded and split into chunks using PyPDFLoader + RecursiveCharacterTextSplitter.
3. Each chunk is converted to an embedding using `sentence-transformers/all-MiniLM-L6-v2`.
4. Embeddings are stored in a FAISS vector index for nearest-neighbour retrieval.
5. When the user asks a question, the app retrieves top-k relevant chunks from FAISS and sends those chunks (and a tightly constrained prompt) to Mistral 7B via the OpenRouter API.
6. The model is instructed to answer only from the provided chunks; if the information is not present, the app explicitly reports that the answer was not found in the document.

Tech stack
----------
- Frontend: Streamlit
- Document loading: LangChain's PyPDFLoader (backed by pypdf)
- Text chunking: RecursiveCharacterTextSplitter (LangChain)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector store: FAISS
- LLM: Mistral 7B (accessed via OpenRouter API)
- Language: Python (recommended 3.8+)
- Optional: python-dotenv for local env management

Features
--------
- Upload a single PDF and build an in-memory FAISS index.
- Chunking and embedding pipeline using proven libraries.
- Retrieval-limited LLM responses: model is prompted to use only retrieved context.
- Explicit "not found in document" replies when the content is not available.
- Lightweight demo suitable for learning and experimentation.

Setup & installation
--------------------
1. Clone the repository:
   - git clone https://github.com/rohitnath-dev/chat-with-pdf.git
2. Create and activate a Python virtual environment:
   - python -m venv .venv
   - source .venv/bin/activate  (Linux / macOS)
   - .venv\Scripts\activate     (Windows)
3. Install dependencies:
   - pip install -r requirements.txt

Example requirements.txt (illustrative)
```
streamlit
langchain
sentence-transformers
faiss-cpu
openrouter
python-dotenv
pypdf
```

4. Provide your OpenRouter API key as an environment variable:
   - export OPENROUTER_API_KEY="your_openrouter_api_key"  (Linux / macOS)
   - setx OPENROUTER_API_KEY "your_openrouter_api_key"   (Windows, new shell)
   - Or add to a `.env` file (if you use python-dotenv)

Important: the OpenRouter API key is required for contacting the remote Mistral 7B model.

How to run the app
------------------
1. Ensure your virtual environment is active and the environment variable is set.
2. From the project root run:
   - streamlit run app.py
3. Open the URL shown by Streamlit (typically http://localhost:8501) in your browser.

Usage instructions
------------------
1. Open the app in your browser.
2. Upload a PDF file.
3. Wait for the app to extract text and build the FAISS index (progress is shown in the UI).
4. Enter a question in the input box and submit.
5. The app retrieves relevant chunks and asks the LLM to answer using only those chunks.
6. If the answer cannot be found within the retrieved chunks, the app will explicitly say that the information is not present in the document.

Behavior and data flow notes:
- The app does not use outside knowledge beyond the uploaded PDF. All answers are produced from the retrieved chunks only.
- Retrieved chunks are sent to OpenRouter to generate responses; therefore, the uploaded document context will be transmitted to the remote LLM endpoint you configure.

Limitations
-----------
- Not production-ready: this is an educational/demo implementation intended for experimentation and learning.
- Dependency on a remote LLM (OpenRouter + Mistral 7B): performance, latency, and cost depend on that service.
- Chunking can lose cross-chunk context; long-form reasoning that requires full-document context may be degraded.
- FAISS uses approximate nearest neighbours; retrieval quality depends on chunk size, embedding quality, and index parameters.
- Privacy: uploaded content and retrieved chunks are sent to the OpenRouter API if you use it—do not upload sensitive or confidential documents unless you accept that.
- Hallucination risk: while the model is instructed to answer only from context, LLMs can still attempt to infer or hallucinate; the system tries to mitigate this by returning "not found" when the evidence is insufficient, but this is not foolproof.

Future improvements
-------------------
- Persist FAISS indices to disk to avoid re-indexing between sessions.
- Support multiple documents and cross-document retrieval.
- Add an evaluation mode (ground-truth QA) for measuring retrieval and answer quality.
- Add streaming responses and partial answer display to reduce perceived latency.
- Add user authentication and access control for multi-user deployment.
- Replace or augment retrieval with hybrid search (keyword + dense vectors) to improve recall.
- Add safety layers and data governance for handling sensitive inputs.

Learning outcome
----------------
By running and reading this project you will gain practical knowledge of:
- Building a simple RAG pipeline end-to-end (document loader → chunking → embeddings → vector DB → LLM).
- How to use sentence-transformers for embeddings and FAISS for similarity search.
- How to constrain LLMs to answer from a context and why that matters to reduce hallucinations.
- How to wire a Streamlit UI to an ML inference pipeline and an external LLM API.
- Trade-offs in chunk size, retrieval quality, latency, and privacy when building RAG systems.

License and contribution
------------------------
This repository is provided as-is for learning and experimentation. Contributions, issues, or suggestions are welcome via pull requests and issues. Do not rely on this code for production use without appropriate review, testing, and security hardening.
