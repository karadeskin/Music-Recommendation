#importing libraries
from flask import Flask, request, jsonify
from recommender import recommend_songs

app = Flask(__name__)

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