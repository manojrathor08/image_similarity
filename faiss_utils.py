import faiss
import numpy as np

# Function to Build FAISS Index
def build_faiss_index(train_embeddings):
    """
    Builds a FAISS index for fast nearest neighbor search.
    """
    dimension = train_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(train_embeddings)
    print(f"FAISS Index built with {index.ntotal} images.")
    return index

# Function to Perform Similarity Search
def search_faiss(index, valid_embeddings, valid_labels, train_labels, k=5, num_queries=150):
    """
    Performs similarity search using FAISS and prints the results.
    """
    random_indices = np.random.choice(len(valid_embeddings), num_queries, replace=False)
    subset_embeddings = valid_embeddings[random_indices]
    subset_labels = valid_labels[random_indices]

    distances, indices = index.search(subset_embeddings, k)

    # Display results
    for i, query_label in enumerate(subset_labels):
        print(f"\nQuery {i + 1}: Label = {query_label}")
        for rank, idx in enumerate(indices[i]):
            print(f"  Rank {rank + 1}: Train Label = {train_labels[idx]}, Distance = {distances[i][rank]}")
