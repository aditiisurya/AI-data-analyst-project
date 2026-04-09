from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Create embedding for query
# -----------------------------
def query_embedding(query):

    embedding = model.encode([query])

    return np.array(embedding)


# -----------------------------
# Vector search using FAISS
# -----------------------------
def search_vector_store(index, query_vector, top_k=5):

    distances, indices = index.search(query_vector, top_k)

    return indices


# -----------------------------
# Retrieve dataset rows
# -----------------------------
def retrieve_rows(indices, text_data):

    results = []

    for i in indices[0]:
        results.append(text_data[i])

    return results


# -----------------------------
# LLM using Google Gemini API
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# -----------------------------
# Generate answer from LLM
# -----------------------------
def generate_answer(context, query):

    prompt = f"""
    You are an AI Data Analyst.

    Use the dataset context below to answer the question.

    Dataset Context:
    {context}

    Question:
    {query}

    Give a clear analytical answer.
    """

    response = llm.invoke(prompt)

    return response.content