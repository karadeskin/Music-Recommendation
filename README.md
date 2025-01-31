# Music Recommendation System

## Overview

This project implements a **content-based music recommendation system** using **PCA (Principal Component Analysis)** and **Nearest Neighbors**. It analyzes song characteristics such as **danceability, energy, tempo, and valence** to recommend similar songs based on a given input track.

## Dataset

The dataset consists of various Spotify songs with audio features extracted from the Spotify API. It includes the following features:
- `danceability`: A measure of how suitable a track is for dancing.
- `energy`: The intensity and activity level of a song.
- `loudness`: The overall volume of the track.
- `speechiness`: The presence of spoken words in a track.
- `acousticness`: The likelihood of a track being acoustic.
- `instrumentalness`: The probability of the track being instrumental.
- `liveness`: The presence of a live audience.
- `valence`: The musical positiveness conveyed by a track.
- `tempo`: The estimated beats per minute (BPM).

The dataset is stored in **`data/spotify_songs.csv`**.

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
python(3) src/app.py
```
Alternatively, you can directly test the recommender:
```bash
python(3) src/recommender.py
```

### Step 2: Get Recommendations

Once the script is running, enter a song name from the dataset to receive a list of recommended tracks.

## Model Evaluation 

The recommendation model is based on content similarity using PCA and cosine distance. The model finds the nearest songs based on their compressed feature representation, making it computationally efficient.

## Future Improvements 

* Improve the recommendation model by incorporating collaborative filtering.
* Expand the dataset with more diverse genres and artists.
* Implement a simple web interface using Flask or Streamlit.

## Acknowledgements 

* Spotify for providing song data via its API.
* scikit-learn for machine learning tools (PCA, Nearest Neighbors).
* pandas & NumPy for data handling and preprocessing.
