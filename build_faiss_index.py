import numpy as np
import faiss
import pickle
import os

# Load your movie embeddings
with open("embeddings/movie_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = np.array(data["embeddings"]).astype("float32")
titles = data["titles"]
imdb_ids = data["imdb_ids"]

# Save mapping for reverse lookup
with open("embeddings/movie_meta.pkl", "wb") as f:
    pickle.dump({"titles": titles, "imdb_ids": imdb_ids}, f)

# Build FAISS index (IVF + LSH)
dimension = embeddings.shape[1]
nlist = 100  # number of clusters
nprobe = 10  # search clusters during query

quantizer = faiss.IndexHNSWFlat(dimension, 32)  # good for LSH hybrid
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# Normalize for inner product similarity
faiss.normalize_L2(embeddings)

# Train index and add vectors
index.train(embeddings)
index.add(embeddings)

# Save the index
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/movie_index.ivf")

print("FAISS IVF index built and saved.")
