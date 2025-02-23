import streamlit as st
import ollama
import chromadb
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def embed_and_store_text(chroma_client, text, doc_id):
    collection = chroma_client.get_or_create_collection(name="rag_documents")
    collection.add(documents=[text], ids=[doc_id])

def retrieve_relevant_docs(chroma_client, query):
    collection = chroma_client.get_or_create_collection(name="rag_documents")
    results = collection.query(query_texts=[query], n_results=3)
    return results["documents"][0] if "documents" in results else []

def query_ollama(model, prompt):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

st.title("Deepseek RAGe")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Model selection
model_name = st.selectbox("Select a DeepSeek model", ["deepseek-r1:1.5b", "deepseek-7b", "deepseek-8b", "deepseek-32b"])

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf", "log"])
if uploaded_file is not None:
    doc_text = extract_text_from_pdf(uploaded_file)
    embed_and_store_text(chroma_client, doc_text, uploaded_file.name)
    st.success("Document indexed successfully!")

query_text = st.text_input("Ask a question based on your documents")
if st.button("Search & Generate Answer"):
    relevant_docs = retrieve_relevant_docs(chroma_client, query_text)
    context = "\n".join(relevant_docs)
    final_prompt = f"Context:\n{context}\n\nQuestion: {query_text}\nAnswer:"
    answer = query_ollama(model_name, final_prompt)
    st.write("**Answer:**", answer)
