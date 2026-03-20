#pip install PyPDF2 faiss-cpu numpy google-generativeai
#pip install -r requirements.txt

import os
import PyPDF2
import faiss
import numpy as np
import google.generativeai as genai

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
genai.configure(api_key="AIzaSyDmEmGjVgYDZ4Y_o6ewABtQgc9UQRd-wE0")
MODEL = genai.GenerativeModel("models//gemini-flash-latest")  # stable model

FILE_PATH = r".\Data\Design Pattern.pdf"
CHUNK_SIZE = 1000  # characters per chunk

# -----------------------------
# 2. READ PDF
# -----------------------------
def read_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# -----------------------------
# 3. CHUNK TEXT
# -----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    start = 0
    print("Chunking text...")
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end      
    
    return chunks

# -----------------------------
# 4. CREATE EMBEDDINGS
# -----------------------------
def embed_text(text_list):
    embeddings = []
    print("Creating embeddings...")
    for t in text_list:
        emb = MODEL.embed_text(t)
        embeddings.append(np.array(emb, dtype='float32'))
    return np.vstack(embeddings)

# -----------------------------
# 5. BUILD VECTOR INDEX (FAISS)
# -----------------------------
def build_index(embeddings):
    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# -----------------------------
# 6. QUERY FUNCTION
# -----------------------------
def query_index(query, chunks, index, top_k=3):
    print("Querying index...")
    query_emb = np.array([MODEL.embed_text(query)], dtype='float32')
    D, I = index.search(query_emb, top_k)
    results = [chunks[i] for i in I[0]]
    
    return "\n".join(results)

# -----------------------------
# 7. ASK QUESTION
# -----------------------------
def ask_question(query, chunks, index):
    print("Asking question...")
    context = query_index(query, chunks, index)
    prompt = f"""
    Answer the question based ONLY on the context below:

    Context:
    {context}

    Question:
    {query}
    """
    response = MODEL.generate_content(prompt)
    return response.text

# -----------------------------
# 8. MAIN EXECUTION
# -----------------------------
document_text = read_pdf(FILE_PATH)
print(f"document_text has {len(document_text)} characters.")
chunks = chunk_text(document_text)
print(f"Created chunk {len(chunks)} chunks.")
embeddings = embed_text(chunks)
print(f"Created embeddings with shape {embeddings.shape}.")
index = build_index(embeddings)
print("Index built successfully.")

query = "What are the key patterns?"
answer = ask_question(query, chunks, index)
print("\nAnswer:\n", answer)