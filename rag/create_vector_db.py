import pandas as pd
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/train.csv")

texts = []

for index, row in df.iterrows():
    text = f"""
    Region: {row['Region']}
    Product: {row['Product Name']}
    Sales: {row['Sales']}
    Date: {row['Order Date']}
    """
    texts.append(text)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
print("Creating embeddings...")
embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Ensure vector store directory exists
os.makedirs("vector_store", exist_ok=True)

# Save FAISS index
faiss.write_index(index, "vector_store/faiss_index")

# Save text data
with open("vector_store/text_data.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Vector database created successfully!")