import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR.parent / "data" / "spotify_songs.csv"
CANDIDATE_FEATURES: List[str] = ["danceability", "energy", "valence"]
DEFAULT_K = 6

def load_dataset(csv_path: Optional[os.PathLike] = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path else DEFAULT_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Place your CSV at {DEFAULT_DATA} or pass a custom path."
        )
    df = pd.read_csv(path)
    if "artists" not in df.columns and "artist_name" in df.columns:
        df = df.rename(columns={"artist_name": "artists"})
    if "track_name" not in df.columns:
        raise ValueError("CSV must include a 'track_name' column.")
    keep_cols = ["track_name", "artists"] + [c for c in CANDIDATE_FEATURES if c in df.columns]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    numeric_feats = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if len(numeric_feats) < 2:
        raise ValueError(
            f"Need at least 2 numeric features, found {numeric_feats}. "
            f"Add more columns or adjust feature list."
        )
    df = df.drop_duplicates(subset=["track_name", "artists"], keep="first").reset_index(drop=True)
    return df

def build_pipeline(n_features: int, n_samples: int) -> Pipeline:
    """
    Build a robust pipeline:
      - Impute means for any missing values (safe if your real data grows later)
      - Standardize
      - PCA only if dimensions allow (>=3 feats and >=4 samples)
      - NearestNeighbors (cosine)
    """
    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
    use_pca = n_features >= 3 and n_samples >= 4
    if use_pca:
        n_components = min(3, n_features, max(1, n_samples - 1))
        steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    steps.append(("knn", NearestNeighbors(n_neighbors=min(DEFAULT_K, n_samples), metric="cosine")))
    return Pipeline(steps)
class Recommender:
    def __init__(self, csv_path: Optional[os.PathLike] = None):
        self.df = load_dataset(csv_path)
        self.features = [c for c in CANDIDATE_FEATURES if c in self.df.columns]
        self.pipeline = build_pipeline(n_features=len(self.features), n_samples=len(self.df))
        X = self.df[self.features].to_numpy()
        Z = self.pipeline.named_steps["imputer"].fit_transform(X)
        Z = self.pipeline.named_steps["scaler"].fit_transform(Z)
        if "pca" in self.pipeline.named_steps:
            Z = self.pipeline.named_steps["pca"].fit_transform(Z)
        self.knn = self.pipeline.named_steps["knn"].fit(Z)
        self.embeddings_ = Z  

    def _find_song_index(self, query: str) -> Optional[int]:
        """Try exact match first, then case-insensitive contains."""
        exact = self.df.index[self.df["track_name"].str.lower() == query.lower()].tolist()
        if exact:
            return exact[0]
        contains = self.df.index[self.df["track_name"].str.contains(query, case=False, na=False)].tolist()
        return contains[0] if contains else None

    def recommend(self, song_name: str, k: int = 5) -> pd.DataFrame:
        if not song_name:
            raise ValueError("song_name must be non-empty.")
        idx = self._find_song_index(song_name)
        if idx is None:
            return pd.DataFrame(columns=["track_name", "artists"])  
        distances, indices = self.knn.kneighbors([self.embeddings_[idx]], n_neighbors=min(k + 1, len(self.df)))
        neighbors = indices[0].tolist()
        if idx in neighbors:
            neighbors.remove(idx)
        recs = self.df.iloc[neighbors][:k][["track_name", "artists"]].reset_index(drop=True)
        return recs

_recommender: Optional[Recommender] = None

def _get_recommender() -> Recommender:
    global _recommender
    if _recommender is None:
        _recommender = Recommender()  
    return _recommender

def recommend_songs(song_name: str, k: int = 5) -> pd.DataFrame | str:
    """
    Returns a DataFrame with columns: track_name, artists, score (0..100)
    or a string error message.
    """
    try:
        R = _get_recommender()
        idx = R._find_song_index(song_name)
        if idx is None:
            return f"Error: '{song_name}' not found in dataset."
        n_query = min(k + 1, len(R.df))
        distances, indices = R.knn.kneighbors([R.embeddings_[idx]], n_neighbors=n_query)
        neighbors = indices[0].tolist()
        if idx in neighbors:
            neighbors.remove(idx)
        dists = distances[0].tolist()
        if len(dists) == n_query and idx in indices[0]:
            dists.pop(neighbors.index(idx)) if idx in neighbors else None
        sims = [max(0.0, (1.0 - (d / 2.0))) for d in dists]  
        sims = sims[:k]
        recs = R.df.iloc[neighbors][:k][["track_name", "artists"]].reset_index(drop=True)
        recs["score"] = (np.array(sims) * 100).round(1)
        return recs
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print(recommend_songs("Blinding Lights"))