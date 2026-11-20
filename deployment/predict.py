from pathlib import Path

import joblib
import numpy as np

from test_feature_maker import song_to_features

# Paths relative to this file, so it works no matter where you run streamlit from
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / ".." / "saved_models" / "best_model.joblib"
MODEL_PATH_NEW = BASE_DIR / ".." / "saved_models" / "tuned_best_model.joblib"

GENRE_NAMES_PATH = BASE_DIR / ".." / "saved_models" / "genre_names.joblib"
GENRE_NAMES_PATH_NEW = BASE_DIR / ".." / "saved_models" / "tunned_genre_names.joblib"

model_old = joblib.load(MODEL_PATH)
genre_names = joblib.load(GENRE_NAMES_PATH)  # was missing in your version -> NameError


model_new = joblib.load(MODEL_PATH_NEW)
genre_names_new = joblib.load(GENRE_NAMES_PATH_NEW)

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


def predict_genre_new(path):
    feats = song_to_features(path)

    # same columns, same order as training
    feats = feats.reindex(columns=model_new.feature_names_in_)

    if hasattr(model_new, "predict_proba"):  # RF, HGB, LR
        proba = model_new.predict_proba(feats).mean(axis=0)
        order = np.argsort(proba)[::-1]
        return [(genre_names[model_new.classes_[i]-1], float(proba[i])) for i in order[:3]]
    else:  # SVC: majority vote
        codes = model_new.predict(feats)
        vals, counts = np.unique(codes, return_counts=True)
        order = np.argsort(counts)[::-1]
        return [(genre_names[vals[i]], float(counts[i] / len(codes))) for i in order[:3]]




if __name__ == "__main__":
    for genre, score in predict_genre_new("../data/test/Reza Pishro -  Ghabrestone Hip Hop.wav"):
        print(f"the music is {genre} with score of {score:.2f}")