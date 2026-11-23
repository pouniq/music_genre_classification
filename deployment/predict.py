from pathlib import Path

import joblib
import numpy as np

from test_feature_maker import song_to_features

# Paths relative to this file, so it works no matter where you run streamlit from
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / ".." / "saved_models" / "best_model.joblib"
MODEL_PATH_NEW = BASE_DIR / ".." / "saved_models" / "tuned_best_model.joblib"


MODEL_PATH_SVC = BASE_DIR / ".." / "saved_models" / "svc_search_tunned.joblib"



GENRE_NAMES_PATH = BASE_DIR / ".." / "saved_models" / "genre_names.joblib"
GENRE_NAMES_PATH_NEW = BASE_DIR / ".." / "saved_models" / "tunned_genre_names.joblib"

model_old = joblib.load(MODEL_PATH)
genre_names = joblib.load(GENRE_NAMES_PATH)  # was missing in your version -> NameError


model_new = joblib.load(MODEL_PATH_NEW)
genre_names_new = joblib.load(GENRE_NAMES_PATH_NEW)


model_svc = joblib.load(MODEL_PATH_SVC)




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




def predict_svc_genre(path, top_k=3):
    feats = song_to_features(path)
    
    classes = model_svc.classes_
    offset = 1   # set to 0 if your training labels start at 0

    if getattr(model_svc[-1], "probability", False):
        scores = model_svc.predict_proba(feats).mean(axis=0)
        order = np.argsort(scores)[::-1][:top_k]
        return [(genre_names[classes[i] - offset], float(scores[i])) for i in order]

    # majority vote over segments
    codes = model_svc.predict(feats)
    vals, counts = np.unique(codes, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k]
    return [(genre_names[vals[i] - offset], float(counts[i] / len(codes))) for i in order]


if __name__ == "__main__":
    for genre, score in predict_svc_genre("../data/test/Chuck Berry - You Never Can Tell.wav"):
        print(f"the music is {genre} with score of {score:.2f}")
        
   
   
feats = song_to_features("../data/test/Chuck Berry - You Never Can Tell.wav")        

expected = list(model_svc.feature_names_in_)   # Pipeline exposes this from the first step
print(len(expected), expected)
print(len(feats.columns), list(feats.columns))
print(set(expected) - set(feats.columns))      # missing at predict time
print(set(feats.columns) - set(expected)) 