__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer




# Load ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="resumes")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieve all stored embeddings
all_data = collection.get(include=["embeddings","metadatas"])

# Extract embeddings and metadata
stored_embeddings = np.array(all_data["embeddings"])
stored_metadata = all_data["metadatas"]

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity manually"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product/(norm_vec1*norm_vec2)
'''
def semantic_search(query, top_n=5):
    """Retrieve the most relevant chunk"""
    # Convert query into an embedding
    query_embedding = embedding_model.encode(query)

    # Compute similarity scores
    similarities = [cosine_similarity(query_embedding, emb) for emb in stored_embeddings]

    # Get indices of top N most similar resumes
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # Return the top matches with metadata
    results = [{"metadata": stored_metadata[i], "similarity": similarities[i]} for i in top_indices]
    return results

# Test the function
#query = "Machine learning engineer with Python experience"
query = "Biology Teacher focus on STEM"
search_results = semantic_search(query)
print(search_results)
'''

def semantic_search(query, top_n=5):
    query_embedding = embedding_model.encode(query)

    similarities = [cosine_similarity(query_embedding, emb) for emb in stored_embeddings]

    top_indices = np.argsort(similarities)[-top_n:][::-1]

    results = [{
        "metadata": stored_metadata[i],
        "similarity": similarities[i],
        "text": stored_metadata[i]["text"]   # Return the actual chunk text
    } for i in top_indices]

    return results

# Test the search function with a sample query
query = "Machine learning engineer with Python experience"
search_results = semantic_search(query)

for result in search_results:
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")
    print()
