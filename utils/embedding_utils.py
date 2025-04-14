import pandas as pd
import os 
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

def generate_movie_embeddings(csv_path="data/movies_cleaned.csv",
                              model_name="all-MiniLM-L6-v2",
                              save_path="embeddings/movie_embeddings.pkl"):
    """
    Generates sentence embeddings for each movie based on title + overview,
    and saves them to disk as a pickle file.
    """
    print("Loading cleaned movie data...")
    df = pd.read_csv(csv_path)

    print("Loading embedding model...")
    model = SentenceTransformer(model_name)

    print("Generating embeddings...")
    texts = (df['title'] + " - " + df['overview']).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    print("Saving embeddings...")
    with open(save_path, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "titles": df["title"].tolist(),
            "imdb_ids": df["imdb_id"].tolist()
        }, f)

    print(f"Embeddings saved to: {save_path}")


def preprocess_movie_metadata(input_path="data/movies_metadata.csv",
                               output_path="data/movies_cleaned.csv",
                               min_overview_length=30):
    """
    Loads and cleans the raw movie metadata file, keeping only
    title, overview, and imdb_id. Filters out invalid entries.
    """
    print("Loading dataset...")
    df = pd.read_csv(input_path, low_memory=False)

    # Keep only necessary columns
    df = df[["title", "overview", "imdb_id"]]

    # Drop rows with missing values
    df.dropna(subset=["title", "overview", "imdb_id"], inplace=True)

    # Filter short overviews
    df = df[df["overview"].str.len() > min_overview_length]

    # Drop duplicate titles
    df.drop_duplicates(subset="title", inplace=True)

    # Keep only valid IMDb IDs (starting with 'tt')
    df = df[df["imdb_id"].astype(str).str.startswith("tt")]

    print(f"Movies after cleaning: {len(df)}")

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
