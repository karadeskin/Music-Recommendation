# Music Recommendation System

## Overview

This project implements a **content-based music recommendation system** using **Principal Component Analysis (PCA)** and **K-Nearest Neighbors (KNN)**.  
It analyzes song characteristics such as **danceability, energy, tempo, and valence** to recommend similar songs based on a given input track.

## **Dataset**
The dataset consists of **pre-collected Spotify song features**.  
It includes the following features:

- **Danceability**: A measure of how suitable a track is for dancing.
- **Energy**: The intensity and activity level of a song.
- **Loudness**: The overall volume of the track.
- **Speechiness**: The presence of spoken words in a track.
- **Acousticness**: The likelihood of a track being acoustic.
- **Instrumentalness**: The probability of the track being instrumental.
- **Liveness**: The presence of a live audience.
- **Valence**: The musical positiveness conveyed by a track.
- **Tempo**: The estimated beats per minute (BPM).

The dataset is stored in **`data/spotify_songs.csv`**.

## Project Structure 
```bash
Music-Recommendation/
│── src/
│   ├── recommender.py      #loads & preprocesses dataset, applies PCA, trains KNN model
│   ├── app.py              #runs the recommendation system via Flask API
│── data/
│   ├── spotify_songs.csv   #dataset with extracted song features
│── requirements.txt        #dependencies for running the project
│── README.md               #project documentation
```

## Files

- `src/recommender.py`: Loads and preprocesses the dataset, applies PCA, trains a Nearest Neighbors model, and provides song recommendations.
- `src/app.py`: Runs the recommendation system and allows users to input a song to get recommendations.
- `data/spotify_songs.csv`: The dataset containing song features.
- `requirements.txt`: Lists required dependencies for running the project.

## Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/karadeskin/Music-Recommendation.git
```

2. Navigate to the project folder:
```bash
cd Music-Recommendation
```

3. Create and activate a virtual environment (optional but recommended):
```bash
python3.10 -m venv venv
source venv/bin/activate  #macOS/Linux
venv\Scripts\activate  #Windows
```

4. Install the required dependencies: 
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Run the Recommendation System 

Run `app.py` to input a song and receive recommendations:
```bash
python3 src/app.py  #macOS/Linux
python src/app.py   #Windows
```
Alternatively, you can directly test the recommender:
```bash
python3 src/recommender.py #macOS/Linux
python src/recommender.py #Windows 
```

### Step 2: Get Recommendations

Once the server is running, open your browser and enter:
```bash
"http://127.0.0.1:5000/recommend?song=Shape of You"
```

## Model Evaluation

The recommendation model is based on content similarity using PCA and Nearest Neighbors with cosine similarity:
* PCA (Principal Component Analysis) reduces dimensionality, improving efficiency while preserving key song features.
* K-Nearest Neighbors (KNN) with cosine similarity is applied to the PCA-transformed feature space to find the most similar songs.

## Future Improvements 

* Improve the recommendation model by incorporating collaborative filtering.
* Expand the dataset with more diverse genres and artists.
* Implement a simple web interface using Flask.
* Deploy the API using platforms like Render, Railway, or Heroku. 

## Acknowledgements 

* Spotify dataset for providing pre-extracted song features.
* scikit-learn for machine learning tools (PCA, Nearest Neighbors).
* pandas & NumPy for data handling and preprocessing.
