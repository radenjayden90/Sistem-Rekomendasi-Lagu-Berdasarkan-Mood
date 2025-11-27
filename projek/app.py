from flask import Flask, render_template, request

import mood_recommender as mr

app = Flask(__name__)

# Load song catalog and features once at startup
songs_df, feat_mat = mr.load_song_catalog("songs_normalize.csv")


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    detected_emotion = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("mood_text", "").strip()
        if user_text:
            # top_k=None -> tampilkan semua rekomendasi yang relevan (tidak dibatasi 10 saja)
            result = mr.recommend_from_text(songs_df, feat_mat, user_text, top_k=None)
            detected_emotion = result["detected_emotion"]
            recommendations = result["recommendations"].to_dict(orient="records")

    return render_template(
        "index.html",
        recommendations=recommendations,
        detected_emotion=detected_emotion,
        user_text=user_text,
    )


if __name__ == "__main__":
    app.run(debug=False)
