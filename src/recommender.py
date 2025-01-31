#importing libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

#locate dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/spotify_songs.csv")
#check if dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"🚨 Dataset not found: {DATA_PATH}. Make sure spotify_songs.csv is in the /data folder.")
#load dataset
df = pd.read_csv(DATA_PATH)
#ensure correct column names
if 'artists' not in df.columns:
    df.rename(columns={'artist_name': 'artists'}, inplace=True)  
#select features
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo']
for col in features:
    if col not in df.columns:
        print(f"⚠️ Warning: Missing column '{col}'. Creating and filling with 0.")
        df[col] = 0  
#handle missing values (fill NaNs with median)
df.fillna(df.median(numeric_only=True), inplace=True)
#get actual available features
available_features = [col for col in features if col in df.columns]
if len(available_features) < 2:
    raise ValueError(f"🚨 Not enough valid features for PCA! Available: {available_features}")
#normalize features
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[available_features])
#dynamically set PCA components (max of available features)
pca_components = min(len(available_features), 3)  
pca = PCA(n_components=pca_components)
df_pca = pca.fit_transform(df_scaled)
#store processed data
df_processed = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(pca_components)])
df_processed['track_name'] = df['track_name']
df_processed['artists'] = df['artists']
#train nearest neighbors model
knn = NearestNeighbors(n_neighbors=min(6, len(df_processed)), metric='cosine')
knn.fit(df_processed.iloc[:, :-2])  

def recommend_songs(song_name):
    """
    given a song title, recommend similar tracks
    """
    song_index = df_processed[df_processed['track_name'].str.lower() == song_name.lower()].index
    if song_index.empty:
        return f"🚨 Error: '{song_name}' not found in dataset. Try another song."
    song_index = song_index[0]
    distances, indices = knn.kneighbors([df_processed.iloc[song_index, :-2]]) 
    recommendations = df_processed.iloc[indices[0][1:], :]
    return recommendations[['track_name', 'artists']]

if __name__ == "__main__":
    print(recommend_songs("Blinding Lights"))  