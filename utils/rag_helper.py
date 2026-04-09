import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import torch
import faiss
import numpy as np

# Load the lightweight model globally for performance
# Reusing the user's preferred model, forced to CPU for deployment stability
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
except Exception as e:
    # Minimal fallback initialization
    embed_model = None
    print(f"Warning: Could not initialize embedding model: {e}")

def extract_text_from_pdf(pdf_file):
    """
    Extracts raw text from an uploaded PDF.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def chunk_text(text, chunk_size=500):
    """
    Splits text into manageable chunks for RAG.
    """
    if not text or len(text.strip()) == 0:
        return []
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    # Ensure all chunks are strings and non-empty
    return [str(c) for c in chunks if str(c).strip()]

def initialize_faiss_index(chunks):
    """
    Computes embeddings for chunks and returns a FAISS index.
    """
    if not chunks or embed_model is None:
        return None
    
    try:
        # Sanitize chunks to prevent TypeError in newer transformers versions
        safe_chunks = [str(c) for c in chunks]
        
        # Pre-compute all embeddings
        embeddings = embed_model.encode(safe_chunks, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype("float32")
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index
    except Exception as e:
        print(f"FAISS Initialization Error: {e}")
        return None

def retrieve_relevant_context(query, chunks, index=None, top_k=2):
    """
    Finds the most relevant text chunks using similarity search.
    Supports FAISS index for high-performance retrieval.
    """
    if not chunks or embed_model is None:
        return ""
    
    try:
        # 1. FAISS High-Speed Path
        if index:
            query_emb = embed_model.encode([str(query)], convert_to_tensor=False)
            query_emb = np.array(query_emb).astype("float32")
            
            distances, indices = index.search(query_emb, top_k)
            relevant_text = "\n\n".join([str(chunks[idx]) for idx in indices[0] if idx != -1 and idx < len(chunks)])
            return relevant_text

        # 2. Legacy/Fallback Path (Dynamic encoding)
        query_emb = embed_model.encode(str(query), convert_to_tensor=True)
        safe_chunks = [str(c) for c in chunks]
        chunk_embs = embed_model.encode(safe_chunks, convert_to_tensor=True)
        
        hits = util.semantic_search(query_emb, chunk_embs, top_k=top_k)
        relevant_text = "\n\n".join([str(chunks[hit['corpus_id']]) for hit in hits[0]])
        return relevant_text
    except Exception as e:
        return f"Retrieval Error: {str(e)}"

def process_knowledge_base(pdf_file):
    """
    Full pipeline to turn a PDF into a searchable chunk list.
    """
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)
    return chunks
