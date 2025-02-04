#importing libraries
from flask import Flask, request, jsonify
from recommender import recommend_songs

#initialize flask app
app = Flask(__name__)

#defining home route 
@app.route('/')
def home():
    return "welcome to my music recommendation API! Use /recommend?song=<song_name> to get recommendations."

#recommend endpt 
@app.route('/recommend', methods=['GET'])
def recommend():
    song_name = request.args.get('song', '').strip()
    if not song_name:
        return jsonify({"error": "No song provided."}), 400
    recommendations = recommend_songs(song_name)
    if isinstance(recommendations, str):  
        return jsonify({"error": recommendations}), 404
    return recommendations.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
