
# Movie Recommendation System with Type-Ahead Search

This project is a production-grade **movie recommendation engine** that mimics the core functionality of **Netflix-style intelligent search**. It combines **type-ahead prefix matching** and **semantic similarity-based recommendations** to help users discover relevant and similar movies with speed and accuracy.

The system leverages a hybrid recommendation architecture:

- **Instant type-ahead search** like Netflix or YouTube — to suggest movies as users type
- **Semantic recommendations** — finding similar movies even when titles are misspelled or abstract
- **Scalable nearest neighbor search** powered by FAISS to handle thousands of movie embeddings efficiently
- **Dynamic poster rendering** with links to IMDb

It is designed to be used in:

- Entertainment platforms (e.g., OTT search engines)
- Movie discovery tools
- AI-powered search demos or GenAI interfaces
- Academic or portfolio-grade machine learning projects

## Demo

![Movie Recommendation Demo](https://github.com/user-attachments/assets/9adbe901-4f09-4824-aa3c-55fbbcd85691)

---

## Dataset

This project is based on the publicly available [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle.

We use the following fields:

- `title`: the name of the movie
- `overview`: the summary of the plot (used for semantic embedding)
- `imdb_id`: used to fetch movie details and link to IMDb
- `release_date`: (optionally usable for filtering or trending features)

The combined `title + overview` is passed through a SentenceTransformer to generate embeddings.

If you add new movies or daily updates via the TMDb API, the system can automatically ingest and update its recommendation index.

---


## Features

- Real-time type-ahead search with fuzzy prefix matching
- Semantic recommendations using transformer-based embeddings
- FAISS IVF+HNSW indexing for efficient similarity search
- Dynamic display of movie posters with IMDb links
- Poster caching using TMDb API with disk persistence
- Configurable API key via environment variables
- Clean 10xN grid layout with uniform poster sizing

---

## Tech Stack

- Backend: Python, Flask
- Embedding Model: SentenceTransformers (`all-MiniLM-L6-v2`)
- Vector Search: FAISS with IVF and HNSW quantizer
- Frontend: HTML, JavaScript, CSS Grid
- External API: TMDb (The Movie Database)

---

## Directory Structure

```
movie-recommender/
├── app.py                          # Flask backend
├── templates/index.html           # Frontend HTML
├── static/style.css               # CSS styling
├── data/poster_cache.json         # Poster URL cache
├── embeddings/movie_embeddings.pkl
├── embeddings/movie_meta.pkl
├── faiss_index/movie_index.ivf
├── .env                           # Environment variables (API key)
├── .gitignore
├── requirements.txt
└── scripts/
    └── build_faiss_index.py       # Script to create FAISS index
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/PavanPapiReddy22/semantic-movie-recommender.git
cd movie-recommender
```

### 2. Create and activate virtual environment

```bash
python -m venv recom
source recom/bin/activate        # On Windows: recom\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```
TMDB_API_KEY=your_tmdb_api_key_here
```

### 5. Prepare Embeddings and FAISS Index

If not already done:

```bash
python scripts/build_faiss_index.py
```

This will:
- Load movie embeddings
- Normalize them
- Create and save a FAISS IVF index
- Save metadata for title/imdb lookup

---

## Running the App

```bash
python app.py
```

Then visit:

```
http://localhost:5000
```

Start typing a movie title in the search bar. The system will:
- Show movies that match the prefix
- Fill the rest with semantically related titles
- Display up to 40 posters (in 10-poster rows)
- Each poster is clickable and redirects to IMDb

---

## TMDb API Key

Register for a free API key at:

[https://www.themoviedb.org/signup](https://www.themoviedb.org/signup)

Use this key in your `.env` file to enable poster fetching.

---

## How It Works

### 1. Embedding Generation

- We use `sentence-transformers` to encode `title + overview` of each movie.
- Embeddings are saved in `movie_embeddings.pkl`.

### 2. FAISS IVF+HNSW Indexing

- An IVF index is built using `IndexIVFFlat` with a `HNSWFlat` quantizer.
- The index is trained, normalized, and saved.
- At query time, we search for top 40 similar vectors using inner product (L2-normalized).

### 3. Prefix Matching

- Before semantic search, we perform a fuzzy-corrected prefix match.
- Prefix matches are prioritized in the results.
- Remaining results are filled with semantic neighbors (FAISS).

### 4. Poster Fetching & Caching

- Poster URLs are fetched from TMDb API.
- They are cached in `poster_cache.json` to avoid repeated API calls.
- A fallback `no-poster.jpg` is used for missing posters.

---

## Future Developments

- Automatic ingestion of new movies from TMDb API daily
- Live update of FAISS index and embedding store
- Dockerization and cloud deployment (Render/EC2)
- Movie detail modals (overview, trailer, rating)
- Personalized recommendations based on watch history
- Multi-language support and advanced filtering (genre, year, etc.)
- Streaming availability integration (Netflix, Prime, etc.)
- LLM-based natural language queries like "funny action movie with robots"

---

## License

This project is licensed under the MIT License.


