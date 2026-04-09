import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import torch
import faiss
import numpy as np

# Load the lightweight model globally for performance
# Reusing the user's preferred model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    """
    Extracts raw text from an uploaded PDF.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    """
    Splits text into manageable chunks for RAG.
    """
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def initialize_faiss_index(chunks):
    """
    Computes embeddings for chunks and returns a FAISS index.
    """
    if not chunks:
        return None
    
    # Pre-compute all embeddings
    embeddings = embed_model.encode(chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index

def retrieve_relevant_context(query, chunks, index=None, top_k=2):
    """
    Finds the most relevant text chunks using similarity search.
    Supports FAISS index for high-performance retrieval.
    """
    if not chunks:
        return ""
    
    # 1. FAISS High-Speed Path
    if index:
        query_emb = embed_model.encode([query], convert_to_tensor=False)
        query_emb = np.array(query_emb).astype("float32")
        
        distances, indices = index.search(query_emb, top_k)
        relevant_text = "\n\n".join([chunks[idx] for idx in indices[0] if idx != -1])
        return relevant_text

    # 2. Legacy/Fallback Path (Dynamic encoding)
    query_emb = embed_model.encode(query, convert_to_tensor=True)
    chunk_embs = embed_model.encode(chunks, convert_to_tensor=True)
    
    hits = util.semantic_search(query_emb, chunk_embs, top_k=top_k)
    relevant_text = "\n\n".join([chunks[hit['corpus_id']] for hit in hits[0]])
    return relevant_text

def process_knowledge_base(pdf_file):
    """
    Full pipeline to turn a PDF into a searchable chunk list.
    """
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)
    return chunks
