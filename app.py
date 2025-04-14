from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import requests
import json
import os
import copy
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import process
from dotenv import load_dotenv


app = Flask(__name__)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS IVF index
faiss_index = faiss.read_index("faiss_index/movie_index.ivf")
faiss_index.nprobe = 10  # You can tune this for speed/recall

# Load metadata: titles + IMDb IDs
with open("embeddings/movie_meta.pkl", "rb") as f:
    meta = pickle.load(f)
    titles = meta["titles"]
    imdb_ids = meta["imdb_ids"]

# TMDb API key
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Poster cache path
CACHE_PATH = "data/poster_cache.json"

# Load or initialize poster cache
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        poster_cache = json.load(f)
else:
    poster_cache = {}


def correct_query(query, choices, threshold=80):
    """Fuzzy correct query for prefix matching."""
    match = process.extractOne(query, choices, score_cutoff=threshold)
    return match[0] if match else query


def get_poster(title):
    if title in poster_cache:
        return poster_cache[title]

    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    try:
        response = requests.get(url).json()
        if response.get("results"):
            poster_path = response["results"][0].get("poster_path")
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            else:
                poster_url = "/static/no-poster.jpg"
        else:
            poster_url = "/static/no-poster.jpg"
    except:
        poster_url = "/static/no-poster.jpg"

    poster_cache[title] = poster_url
    return poster_url


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def search():
    query_raw = request.args.get("q", "").strip()
    if not query_raw:
        return jsonify({"results": []})

    # Prefix match with fuzzy correction
    query_corrected = correct_query(query_raw, titles)

    # Prefix matches first (limit to 16)
    prefix_results = []
    seen_titles = set()
    for i, title in enumerate(titles):
        if title.lower().startswith(query_corrected.lower()):
            prefix_results.append(
                {
                    "title": title,
                    "imdb_id": imdb_ids[i],
                    "poster": get_poster(title),
                    "url": f"https://www.imdb.com/title/{imdb_ids[i]}",
                }
            )
            seen_titles.add(title)
        if len(prefix_results) >= 16:
            break

    # Semantic search using FAISS (limit to 40 - len(prefix_results))
    query_embed = model.encode([query_raw])
    query_embed = np.array(query_embed).astype("float32")
    faiss.normalize_L2(query_embed)

    _, top_indices = faiss_index.search(query_embed, 40)
    top_indices = top_indices[0]

    semantic_results = []
    for idx in top_indices:
        title = titles[idx]
        if title in seen_titles:
            continue
        semantic_results.append(
            {
                "title": title,
                "imdb_id": imdb_ids[idx],
                "poster": get_poster(title),
                "url": f"https://www.imdb.com/title/{imdb_ids[idx]}",
            }
        )
        if len(prefix_results) + len(semantic_results) >= 40:
            break

    # Combine: prefix results first, then semantic
    final_results = prefix_results + semantic_results

    # Save updated poster cache once per request
    try:
        poster_cache_copy = copy.deepcopy(poster_cache)
        with open(CACHE_PATH, "w") as f:
            json.dump(poster_cache_copy, f)
    except Exception as e:
        print("Error writing cache:", e)

    return jsonify({"results": final_results})


if __name__ == "__main__":
    app.run(debug=True)
