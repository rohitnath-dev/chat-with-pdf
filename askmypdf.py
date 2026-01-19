import os
import tempfile
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage


os.environ["OPENROUTER_API_KEY"] = ""  # your api key here

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Chat with Your PDF",
    },
)

st.title("Chat with Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

retriever = None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    if chunks:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

user_input = st.chat_input("Ask a question about the PDF")

if user_input:
    with st.chat_message("user"):
        st.text(user_input)

    if retriever is None:
        st.warning("Please upload a PDF first.")
    else:
        docs = retriever.invoke(user_input)
        context = "\n\n".join([d.page_content for d in docs])

        if not context.strip():
            answer = "I could not find the answer in the uploaded document."
        else:
            prompt = f"""
You are a careful and reliable assistant whose job is to answer questions
ONLY using the information provided in the context below.

Rules you MUST follow:
1. Use ONLY the given context to answer the question.
2. Do NOT use any outside knowledge, assumptions, or prior training.
3. If the answer is not clearly present in the context, say:
   "I could not find the answer in the uploaded document."
4. Do NOT hallucinate or make up details.
5. Keep the answer clear, structured, and easy to understand.
6. If the user asks for a summary, provide a concise but complete summary
   based strictly on the context.
7. If the question is factual, answer precisely.
8. If the question is conceptual, explain it clearly using only the context.

Context (from the uploaded PDF):
--------------------------------
{context}
--------------------------------

User Question:
{user_input}

Now generate the best possible answer strictly following the rules above.
"""
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

        with st.chat_message("assistant"):
            st.text(answer)
