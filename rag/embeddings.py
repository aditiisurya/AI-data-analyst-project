from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_list):
    embeddings = model.encode(text_list)
    return embeddings