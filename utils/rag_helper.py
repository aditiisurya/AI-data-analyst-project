import os
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import torch
import faiss
import numpy as np

# --- EMBEDDING MODEL SETUP ---
# We use 'all-MiniLM-L6-v2', a lightweight and efficient model for sentence embeddings.
# Attempt to use GPU (CUDA) for acceleration, otherwise fallback to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
except Exception as e:
    embed_model = None
    print(f"Warning: Could not initialize embedding model: {e}")

def extract_text_from_pdf(pdf_file):
    """
    Step 1: Raw Text Extraction.
    Uses 'pypdf' to read through all pages of a PDF and concatenate text.
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
    Step 2: Semantic Chunking.
    Breaks large documents into smaller overlapping or fixed-size segments 
    to fit within LLM context windows and improve retrieval accuracy.
    """
    if not text or len(text.strip()) == 0:
        return []
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [str(c) for c in chunks if str(c).strip()]

def initialize_faiss_index(chunks):
    """
    Step 3: .
    Turns text chunks into numerical vectors (embeddings) and stores them 
    in a FAISS index for ultra-fast similarity searches.
    """
    if not chunks or embed_model is None:
        return None
    
    try:
        safe_chunks = [str(c) for c in chunks]
        embeddings = embed_model.encode(safe_chunks, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype("float32")
        
        # IndexFlatL2 uses Euclidean distance for similarity
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index
    except Exception as e:
        print(f"FAISS Initialization Error: {e}")
        return None

def convert_df_to_text(df):
    """
    Converts a DataFrame into a list of text chunks using optimized vectorized operations.
    Each chunk represents a row in a human-readable format.
    """
    if df is None or df.empty:
        return []
    
    # Use a faster apply method instead of iterrows
    def format_row(row):
        items = [f"{col} is {val}" for col, val in row.items() if pd.notna(val)]
        return "A data record where " + ", ".join(items) + "."
    
    return df.apply(format_row, axis=1).tolist()

def retrieve_relevant_context(query, chunks, index=None, top_k=50):
    """
    Step 4: Vector Retrieval.
    Uses the FAISS index to find the 'top_k' most similar chunks to the query.
    If no index is provided, it falls back to a slower dynamic semantic search.
    """
    if not chunks or embed_model is None:
        return ""
    
    try:
        # Path A: High-Speed Index Search (Preferred)
        if index:
            query_emb = embed_model.encode([str(query)], convert_to_tensor=False)
            query_emb = np.array(query_emb).astype("float32")
            
            # Ensure top_k doesn't exceed number of chunks
            search_k = min(top_k, len(chunks))
            distances, indices = index.search(query_emb, search_k)
            # Fetch the actual text for the indices returned by FAISS
            relevant_text = "\n\n".join([str(chunks[idx]) for idx in indices[0] if idx != -1 and idx < len(chunks)])
            return relevant_text

        # Path B: Dynamic fallback (Direct comparison)
        query_emb = embed_model.encode(str(query), convert_to_tensor=True)
        safe_chunks = [str(c) for c in chunks]
        chunk_embs = embed_model.encode(safe_chunks, convert_to_tensor=True)
        
        search_k = min(top_k, len(chunks))
        hits = util.semantic_search(query_emb, chunk_embs, top_k=search_k)
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
