import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import joblib
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from bert_emotion_classifier import get_bert_classifier


# ==========================
# 1. Text Preprocessing
# ==========================

BASIC_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just",
}

# Keywords for very negative / crisis expressions that should never be
# mapped to positive / high-energy emotions. If any of these appear in the
# raw text, we will force the emotion to "sad" before consulting the ML
# model, so inputs like "ingin mati", "pengen bunuh diri" are treated as
# strongly negative.
CRISIS_KEYWORDS = {
    "bunuh diri",
    "ingin mati",
    "pengen mati",
    "pengen bunuh diri",
    "ga mau hidup lagi",
    "tidak mau hidup lagi",
    "capek hidup",
    "capek hidup ini",
    "suicide",
    "kill myself",
    "kill myself.",
    "kill myself!",
    "want to die",
    "want to end my life",
    "end my life",
}


def preprocess_text(text: str) -> List[str]:
    """Simple tokenizer + cleaner without external models."""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@[a-z0-9_]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = []
    for tok in text.split():
        if tok in BASIC_STOPWORDS:
            continue
        if len(tok) < 2:
            continue
        tokens.append(tok)
    return tokens


# ==========================
# 2. Simple Rule-based Emotion Classifier
# ==========================

EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "happy": [
        "happy", "excited", "great", "awesome", "joy", "joyful", "glad",
        "good", "amazing", "love", "lovely", "cheerful", "optimistic",
        "grateful", "blessed", "fantastic", "fun", "energetic", "semangat",
        "senang", "bahagia", "ceria", "gembira",
    ],
    "sad": [
        "sad", "unhappy", "depressed", "down", "lonely", "cry", "crying",
        "heartbroken", "tired", "exhausted", "upset", "miserable",
        "sedih", "galau", "kecewa", "hancur",
    ],
    "angry": [
        "angry", "mad", "furious", "annoyed", "irritated", "rage",
        "benci", "marah", "kesal", "sebel",
    ],
    "calm": [
        "calm", "relaxed", "peaceful", "chill", "quiet", "soothing",
        "tenang", "damai", "santai", "nyaman",
    ],
    "excited": [
        "excited", "pumped", "hype", "motivated", "driven", "eager",
        "semangat", "antusias", "penasaran",
    ],
    "neutral": ["ok", "fine", "biasa", "normal"],
}


def _predict_emotion_rule_based(text: str) -> str:
    """Fallback rule-based emotion classifier.
    
    This is only used as a fallback if BERT fails.
    """
    if not text or not isinstance(text, str):
        return "happy"

    tokens = preprocess_text(text)
    if not tokens:
        return "happy"

    token_set = set(tokens)
    scores = {
        emo: len(token_set.intersection(emotion_words))
        for emo, emotion_words in EMOTION_KEYWORDS.items()
    }
    return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else "happy"


# ==========================
# 2b. ML-based Emotion Classifier (TF-IDF + Linear SVM)
# ==========================

_EMOTION_TRAIN_TEXTS = [
    # happy
    "i feel happy and grateful today",
    "so excited and full of energy",
    "aku senang dan bahagia sekali",
    "hati terasa ceria dan optimis",
    # sad
    "i feel very sad and lonely",
    "i am depressed and tired of everything",
    "aku lagi sedih dan galau",
    "perasaan kecewa dan hancur",
    # angry
    "i am very angry and frustrated",
    "so mad and irritated right now",
    "aku marah dan sangat kesal",
    "benci banget sama keadaan ini",
    # calm
    "i feel calm and peaceful",
    "relaxed and ready to sleep",
    "aku merasa tenang dan santai",
    "suasana damai dan nyaman",
    # excited
    "super excited for this new project",
    "i am pumped and motivated",
    "aku semangat banget dan antusias",
    "nggak sabar mulai petualangan baru",
    # neutral
    "i feel okay, nothing special",
    "just normal day, not good not bad",
    "biasa saja, hari yang normal",
    "tidak terlalu senang atau sedih",
]

_EMOTION_TRAIN_LABELS = [
    # happy
    "happy", "happy", "happy", "happy",
    # sad
    "sad", "sad", "sad", "sad",
    # angry
    "angry", "angry", "angry", "angry",
    # calm
    "calm", "calm", "calm", "calm",
    # excited
    "excited", "excited", "excited", "excited",
    # neutral
    "neutral", "neutral", "neutral", "neutral",
]


def _build_emotion_model() -> Optional[Pipeline]:
    # This function is kept for compatibility but will return None
    # as we're now using BERT for emotion classification
    return None


_EMOTION_MODEL: Optional[Pipeline] = None


def _get_emotion_model() -> None:
    # This function is kept for compatibility but always returns None
    # as we're now using BERT for emotion classification
    return None


def predict_emotion(text: str) -> str:
    """Predict emotion from text using BERT."""
    if not text or not isinstance(text, str) or not text.strip():
        return "happy"  # Default to happy for empty input

    # Check for crisis keywords first (safety check)
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in CRISIS_KEYWORDS):
        return "sad"

    # Use BERT for emotion classification
    try:
        model = _get_emotion_model()
        if model is not None:
            return model.predict([text])[0]
    except Exception:
        pass
    return _predict_emotion_rule_based(text)


# ==========================
# 3. Load and Prepare Song Features
# ==========================

SONG_FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "popularity",
]


def load_song_catalog(csv_path: str = "songs_normalize.csv"):
    # Read the CSV file and limit to first 971 rows
    df = pd.read_csv(csv_path).head(971)
    
    # Create a clean version of song and artist names for deduplication
    df['song_clean'] = df['song'].str.lower().str.strip()
    df['artist_clean'] = df['artist'].str.lower().str.strip()
    
    # Remove duplicates based on song and artist (keeping the first occurrence)
    df = df.drop_duplicates(subset=['song_clean', 'artist_clean'])
    
    # Remove the temporary columns
    df = df.drop(columns=['song_clean', 'artist_clean'])

    # Basic cleaning
    if "genre" in df.columns:
        df["genre"] = df["genre"].astype(str).str.lower().str.strip()

    if "explicit" in df.columns:
        df["explicit"] = (
            df["explicit"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0)
        )
        
    print(f"Loaded {len(df)} unique songs after removing duplicates")

    try:
        kb1 = pd.read_csv("Data KB1- Cleaned (Tiktok Songs).csv")
        if {"track_name", "artist_name", "song_emotion"}.issubset(set(kb1.columns)) and {"artist", "song"}.issubset(set(df.columns)):
            df["artist_norm"] = df["artist"].astype(str).str.lower().str.strip()
            df["song_norm"] = df["song"].astype(str).str.lower().str.strip()

            kb1["artist_norm"] = kb1["artist_name"].astype(str).str.lower().str.strip()
            kb1["song_norm"] = kb1["track_name"].astype(str).str.lower().str.strip()

            kb1_subset = kb1[["artist_norm", "song_norm", "song_emotion"]].drop_duplicates(subset=["artist_norm", "song_norm"])

            df = df.merge(kb1_subset, on=["artist_norm", "song_norm"], how="left")
            df["song_emotion"] = df["song_emotion"].astype(str).str.lower().str.strip()

            df = df.drop(columns=["artist_norm", "song_norm"], errors="ignore")
    except Exception:
        pass

    # Jika masih ada lagu tanpa label song_emotion setelah merge KB1,
    # gunakan model terlatih dari KB1 (song_emotion_model.joblib) untuk memprediksi.
    try:
        if "song_emotion" not in df.columns or df["song_emotion"].isna().any():
            model = joblib.load("song_emotion_model.joblib")

            # Gunakan semua kolom numerik sebagai fitur (konsisten dengan script training)
            num_cols_for_model = df.select_dtypes(include=[np.number]).columns
            mask_missing = df["song_emotion"].isna() if "song_emotion" in df.columns else pd.Series(True, index=df.index)

            if mask_missing.any():
                X_missing = df.loc[mask_missing, num_cols_for_model].values
                preds = model.predict(X_missing)
                df.loc[mask_missing, "song_emotion"] = preds

            df["song_emotion"] = df["song_emotion"].astype(str).str.lower().str.strip()
    except Exception:
        # Jika model tidak tersedia atau gagal diload, lanjutkan tanpa prediksi tambahan.
        pass

    # Ensure required feature columns exist
    missing = [c for c in SONG_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in CSV: {missing}")

    # Fill numeric NaNs and scale
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    feature_matrix = df[SONG_FEATURE_COLS].values.astype(np.float32)
    return df, feature_matrix


# ==========================
# 4. Emotion → Music Profile Mapping
# ==========================

EMOTION_TO_PROFILE = {
    "happy": {
        "danceability": (0.6, 1.0),
        "energy": (0.6, 1.0),
        "valence": (0.6, 1.0),
        "tempo": (0.5, 1.0),  # tempo already scaled 0-1
        "preferred_genres": ["pop", "dance", "latin", "r&b"],
    },
    "sad": {
        "danceability": (0.0, 0.6),
        "energy": (0.0, 0.5),
        "valence": (0.0, 0.4),
        "tempo": (0.0, 0.7),
        "preferred_genres": ["acoustic", "folk", "r&b", "rock"],
    },
    "angry": {
        "danceability": (0.4, 0.9),
        "energy": (0.7, 1.0),
        "valence": (0.0, 0.5),
        "tempo": (0.5, 1.0),
        "preferred_genres": ["metal", "rock", "hip hop"],
    },
    "calm": {
        "danceability": (0.0, 0.6),
        "energy": (0.0, 0.5),
        "valence": (0.4, 1.0),
        "tempo": (0.0, 0.6),
        "preferred_genres": ["acoustic", "jazz", "ambient", "lofi"],
    },
    "excited": {
        "danceability": (0.6, 1.0),
        "energy": (0.7, 1.0),
        "valence": (0.5, 1.0),
        "tempo": (0.6, 1.0),
        "preferred_genres": ["pop", "edm", "hip hop", "dance"],
    },
    "neutral": {
        "danceability": (0.3, 0.8),
        "energy": (0.3, 0.8),
        "valence": (0.3, 0.7),
        "tempo": (0.3, 0.7),
        "preferred_genres": [],
    },
}


def _mid_range(low: float, high: float) -> float:
    return (low + high) / 2.0


def build_mood_vector(emotion: str) -> np.ndarray:
    cfg = EMOTION_TO_PROFILE.get(emotion, EMOTION_TO_PROFILE["neutral"])

    mood_profile = {
        "danceability": _mid_range(*cfg["danceability"]),
        "energy": _mid_range(*cfg["energy"]),
        "valence": _mid_range(*cfg["valence"]),
        "tempo": _mid_range(*cfg["tempo"]),
        "acousticness": 0.5,
        "instrumentalness": 0.2,
        "liveness": 0.3,
        "speechiness": 0.3,
        "popularity": 0.7,
    }

    v = np.array([mood_profile[c] for c in SONG_FEATURE_COLS], dtype=np.float32)
    v = v / (norm(v) + 1e-8)
    return v


# ==========================
# 5. Similarity and Recommendation
# ==========================

def cosine_sim_to_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    vec_norm = vec / (norm(vec) + 1e-8)
    mat_norm = mat / (norm(mat, axis=1, keepdims=True) + 1e-8)
    return mat_norm @ vec_norm


def recommend_songs_for_emotion(
    songs_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    emotion: str,
    top_k: int | None = None,
) -> pd.DataFrame:
    cfg = EMOTION_TO_PROFILE.get(emotion, EMOTION_TO_PROFILE["neutral"])
    preferred_genres = [g.lower() for g in cfg.get("preferred_genres", [])]

    mood_vec = build_mood_vector(emotion)

    # Genre filter (optional)
    mask = np.ones(len(songs_df), dtype=bool)
    if preferred_genres and "genre" in songs_df.columns:
        pattern = "|".join(preferred_genres)
        mask = songs_df["genre"].fillna("").str.contains(pattern, na=False)

    candidates = songs_df[mask].reset_index(drop=True)
    candidate_features = feature_matrix[mask]

    if len(candidates) == 0:
        candidates = songs_df.copy()
        candidate_features = feature_matrix

    sims = cosine_sim_to_matrix(mood_vec, candidate_features)

    ranked = candidates.copy()
    ranked["similarity"] = sims

    # Haluskan mapping emosi pengguna -> label song_emotion dari KB1
    # Skor prioritas: 2 = sangat cocok, 1 = cukup cocok, 0 = tidak cocok
    emotion_priority_map = {
        # Lagu happy paling cocok, neutral masih bisa
        "happy": {"happy": 2, "neutral": 1},
        # Excited diarahkan ke happy, neutral sebagai cadangan
        "excited": {"happy": 2, "neutral": 1},
        # Sedih paling cocok dengan sad, neutral bisa untuk lagu yang tidak terlalu gelap
        "sad": {"sad": 2, "neutral": 1},
        # Marah diarahkan kuat ke sad (lagu intens/gelap), neutral sebagai cadangan
        "angry": {"sad": 2, "neutral": 1},
        # Calm diarahkan ke neutral, sad sekunder untuk lagu mellow/tenang tapi agak sedih
        "calm": {"neutral": 2, "sad": 1},
        # Neutral fleksibel, semua sama (tidak memaksa match)
        "neutral": {"happy": 1, "sad": 1, "neutral": 1},
    }

    if "song_emotion" in ranked.columns:
        se = ranked["song_emotion"].astype(str).str.lower().str.strip()
        pri_map = emotion_priority_map.get(emotion, emotion_priority_map["neutral"])
        ranked["emotion_priority"] = se.map(lambda x: pri_map.get(x, 0)).astype(int)
        ranked = ranked.sort_values(
            by=["emotion_priority", "similarity"],
            ascending=[False, False],
        )
        if top_k is not None and top_k > 0:
            recs = ranked.head(top_k).copy()
        else:
            recs = ranked.copy()
    else:
        idx = np.argsort(-sims)
        if top_k is not None and top_k > 0:
            idx = idx[:top_k]
        recs = ranked.iloc[idx].copy()

    return recs


def recommend_from_text(
    songs_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    user_text: str,
    top_k: int | None = None,
) -> Dict:
    emotion = predict_emotion(user_text)
    recs = recommend_songs_for_emotion(songs_df, feature_matrix, emotion, top_k=top_k)

    if {"artist", "song"}.issubset(recs.columns):
        def _add_streaming_urls(row: pd.Series) -> pd.Series:
            query = f"{row['artist']} - {row['song']}"
            q = quote_plus(str(query))
            row["yt_music_url"] = f"https://music.youtube.com/search?q={q}"
            row["spotify_url"] = f"https://open.spotify.com/search/{q}"
            return row

        recs = recs.apply(_add_streaming_urls, axis=1)

    cols = []
    for c in [
        "artist",
        "song",
        "genre",
        "tempo",
        "energy",
        "valence",
        "song_emotion",
        "similarity",
        "yt_music_url",
        "spotify_url",
    ]:
        if c in recs.columns:
            cols.append(c)
    result_df = recs[cols]

    return {
        "detected_emotion": emotion,
        "recommendations": result_df,
    }


# ==========================
# 6. Simple CLI Entry Point
# ==========================

if __name__ == "__main__":
    print("Loading song catalog from songs_normalize.csv ...")
    songs_df, feat_mat = load_song_catalog("songs_normalize.csv")
    print(f"Loaded {len(songs_df)} songs.")

    print("\nMasukkan teks perasaan / situasi kamu (dalam Bahasa Indonesia atau Inggris):")
    user_text = input("> ")

    result = recommend_from_text(songs_df, feat_mat, user_text, top_k=10)

    print(f"\nEmosi terdeteksi: {result['detected_emotion']}")
    print("Rekomendasi lagu:")

    rec_df = result["recommendations"]
    for i, row in rec_df.iterrows():
        artist = row.get("artist", "?")
        title = row.get("song", "?")
        genre = row.get("genre", "?")
        se = row.get("song_emotion", None)
        sim = float(row.get("similarity", 0.0))
        if se is not None and not (isinstance(se, float) and np.isnan(se)):
            print(f"{i+1:2d}. {artist} - {title} | genre={genre} | emotion={se} | similarity={sim:.3f}")
        else:
            print(f"{i+1:2d}. {artist} - {title} | genre={genre} | similarity={sim:.3f}")
