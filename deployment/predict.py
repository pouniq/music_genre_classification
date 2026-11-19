from pathlib import Path

import joblib
import numpy as np

from test_feature_maker import song_to_features

# Paths relative to this file, so it works no matter where you run streamlit from
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / ".." / "saved_models" / "best_model.joblib"
GENRE_NAMES_PATH = BASE_DIR / ".." / "saved_models" / "genre_names.joblib"

model_old = joblib.load(MODEL_PATH)
genre_names = joblib.load(GENRE_NAMES_PATH)  # was missing in your version -> NameError


def predict_genre(path):
    feats = song_to_features(path)

    # same columns, same order as training
    feats = feats.reindex(columns=model_old.feature_names_in_)

    if hasattr(model_old, "predict_proba"):  # RF, HGB, LR
        proba = model_old.predict_proba(feats).mean(axis=0)
        order = np.argsort(proba)[::-1]
        return [(genre_names[model_old.classes_[i]], float(proba[i])) for i in order[:3]]
    else:  # SVC: majority vote
        codes = model_old.predict(feats)
        vals, counts = np.unique(codes, return_counts=True)
        order = np.argsort(counts)[::-1]
        return [(genre_names[vals[i]], float(counts[i] / len(codes))) for i in order[:3]]


if __name__ == "__main__":
    for genre, score in predict_genre("../data/test/Reza Pishro -  Ghabrestone Hip Hop.wav"):
        print(f"the music is {genre} with score of {score:.2f}")