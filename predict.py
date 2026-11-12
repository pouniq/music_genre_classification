import joblib
import numpy as np
from audio_features.test_feature_maker import song_to_features

model_old = joblib.load("saved_models/best_model.joblib")
model_tunned = joblib.load("saved_models/tuned_best_model.joblib")


genre_names = joblib.load("saved_models/genre_names.joblib")
genre_names_tunned = joblib.load("saved_models/tunned_genre_names.joblib")


def predict_song(path):
    feats = song_to_features(path)

    # same columns, same order as training
    feats = feats.reindex(columns=model_old.feature_names_in_)

    if hasattr(model_old, "predict_proba"):             # RF, HGB, LR
        proba = model_old.predict_proba(feats).mean(axis=0)
        order = np.argsort(proba)[::-1]
        return [(genre_names[model_old.classes_[i]], float(proba[i])) for i in order[:3]]
    else:                                            # SVC: majority vote
        codes = model_old.predict(feats)
        vals, counts = np.unique(codes, return_counts=True)
        order = np.argsort(counts)[::-1]
        return [(genre_names[vals[i]], counts[i] / len(codes)) for i in order[:3]]

if __name__ == "__main__":
    for genre, score in predict_song("data/test/Clipse.wav"):
        print(f'the music is {genre} with score of {score:.2f}')
        
        
        
        
        
def predict_song(path):
    feats = song_to_features(path)

    # same columns, same order as training
    feats = feats.reindex(columns=model_tunned.feature_names_in_)

    if hasattr(model_tunned, "predict_proba"):             # RF, HGB, LR
        proba = model_tunned.predict_proba(feats).mean(axis=0)
        order = np.argsort(proba)[::-1]
        return [(genre_names[model_tunned.classes_[i]-1], float(proba[i])) for i in order[:3]]
    else:                                            # SVC: majority vote
        codes = model_tunned.predict(feats)
        vals, counts = np.unique(codes, return_counts=True)
        order = np.argsort(counts)[::-1]
        return [(genre_names[vals[i]], counts[i] / len(codes)) for i in order[:3]]

if __name__ == "__main__":
    for genre, score in predict_song("data/test/Clipse.wav"):
        print(f'the music is {genre} with score of {score:.2f}')