from flask import Flask, request, jsonify, render_template_string
from recommender import recommend_songs, _get_recommender

app = Flask(__name__)

@app.route('/')
def home():
    return "welcome to my music recommendation API! Use /recommend?song=<song_name>&k=5"

@app.route('/songs')
def songs():
    r = _get_recommender()
    items = r.df[["track_name", "artists"]].to_dict(orient="records")
    return jsonify(items)

@app.route('/recommend', methods=['GET'])
def recommend():
    song_name = request.args.get('song', '').strip()
    k = int(request.args.get('k', 5))
    if not song_name:
        return jsonify({"error": "No song provided."}), 400
    res = recommend_songs(song_name, k=k)
    if isinstance(res, str):
        return jsonify({"error": res}), 404
    return jsonify(res.to_dict(orient='records'))

PRETTY_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Music Recs</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px; }
    h1 { margin: 0 0 16px; }
    form { margin-bottom: 16px; display:flex; gap:8px; }
    input, button, select { padding: 8px 10px; font-size: 16px; }
    table { border-collapse: collapse; width: 100%; margin-top: 12px; }
    th, td { border-bottom: 1px solid #eee; padding: 10px; text-align: left; }
    .score { font-variant-numeric: tabular-nums; }
  </style>
</head>
<body>
  <h1>Music Recommendation Demo</h1>
  <form action="/recommend/ui" method="get">
    <input name="song" placeholder="Enter a song (e.g., Blinding Lights)" value="{{song|default('')}}" style="flex:1" />
    <select name="k">
      {% for n in [3,5,10] %}
        <option value="{{n}}" {% if k == n %}selected{% endif %}>Top {{n}}</option>
      {% endfor %}
    </select>
    <button>Recommend</button>
  </form>
  {% if error %}
    <p style="color:#b00020">{{ error }}</p>
  {% elif rows %}
    <table>
      <thead><tr><th>#</th><th>Track</th><th>Artist</th><th>Score</th></tr></thead>
      <tbody>
        {% for r in rows %}
          <tr>
            <td>{{ loop.index }}</td>
            <td>{{ r.track_name }}</td>
            <td>{{ r.artists }}</td>
            <td class="score">{{ "%.1f"|format(r.score) }}%</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  {% endif %}
</body>
</html>
"""

@app.route('/recommend/ui')
def recommend_ui():
    song = request.args.get('song', '').strip()
    k = int(request.args.get('k', 5))
    rows, error = [], None
    if song:
        res = recommend_songs(song, k=k)
        if isinstance(res, str):
            error = res
        else:
            rows = res.to_dict(orient="records")
    return render_template_string(PRETTY_HTML, rows=rows, error=error, song=song, k=k)

if __name__ == '__main__':
    app.run(debug=False)
