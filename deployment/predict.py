from pathlib import Path

import joblib
import numpy as np

from test_feature_maker import song_to_features

# Paths relative to this file, so it works no matter where you run streamlit from
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / ".." / "saved_models" / "best_model.joblib"
MODEL_PATH_NEW = BASE_DIR / ".." / "saved_models" / "tuned_best_model.joblib"

MODEL_PATH_SVC_BASE = BASE_DIR / ".." / "saved_models" / "svc_model_raw.joblib"
MODEL_PATH_SVC = BASE_DIR / ".." / "saved_models" / "svc_search_tunned.joblib"



GENRE_NAMES_PATH = BASE_DIR / ".." / "saved_models" / "genre_names.joblib"
GENRE_NAMES_PATH_NEW = BASE_DIR / ".." / "saved_models" / "tunned_genre_names.joblib"

model_old = joblib.load(MODEL_PATH)
genre_names = joblib.load(GENRE_NAMES_PATH)  # was missing in your version -> NameError


model_new = joblib.load(MODEL_PATH_NEW)
genre_names_new = joblib.load(GENRE_NAMES_PATH_NEW)

model_svc_base = joblib.load(MODEL_PATH_SVC_BASE)
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




def _predict(model, path, names, top_k=3):
    feats = song_to_features(path)
    feats = feats[list(model.feature_names_in_)]      # drops ber1_*, fails loudly if a feature is missing

    classes = model.classes_
    if np.issubdtype(classes.dtype, np.integer):      # integer labels: 0-based or 1-based
        offset = int(classes.min())
        to_name = lambda c: names[int(c) - offset]
    else:                                             # labels are already genre names
        to_name = str

    if hasattr(model, "predict_proba"):               # SVC trained with probability=True
        proba = model.predict_proba(feats).mean(axis=0)
        order = np.argsort(proba)[::-1][:top_k]
        return [(to_name(classes[i]), float(proba[i])) for i in order]

    pred = model.predict(feats)                       # otherwise: majority vote over segments
    vals, counts = np.unique(pred, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k]
    return [(to_name(vals[i]), float(counts[i] / len(pred))) for i in order]


def predict_svc_base_genre(path, top_k=3):
    return _predict(model_svc_base, path, genre_names, top_k)

def predict_svc_genre(path, top_k=3):
    return _predict(model_svc, path, genre_names_new, top_k)   # use genre_names if that's what it was trained with



if __name__ == "__main__":
    for genre, score in predict_genre("../data/test/Future - Trap Niggas.wav"):
        print(f"the music is {genre} with score of {score:.2f}")
        
   
   

