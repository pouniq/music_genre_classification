import io
import os
import tempfile

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from predict import predict_genre, predict_genre_new, predict_svc_genre, predict_svc_base_genre

CLIP_SEC = 30

st.set_page_config(page_title="Song Genre Classification", page_icon="🎵", layout="centered")


@st.cache_data(show_spinner="Loading audio...")
def load_audio(file_bytes: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
    finally:
        os.remove(path)
    return y, sr


def plot_waveform(y, sr, start, end):
    # downsample for plotting speed
    step = max(1, len(y) // 5000)
    t = np.arange(0, len(y), step) / sr
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(t, y[::step], linewidth=0.5)
    ax.axvspan(start, end, alpha=0.3, color="tab:orange")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    fig.tight_layout()
    return fig


st.title("Song Genre Classification")
st.write("Upload a song, choose a 30-second part & click **Predict genre**")
st.subheader("Audio file")

uploaded = st.file_uploader(
    "Choose an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"]
)

if uploaded is not None:
    suffix = os.path.splitext(uploaded.name)[1]
    y, sr = load_audio(uploaded.getvalue(), suffix)
    duration = len(y) / sr

    # ---- segment selection ----
    if duration <= CLIP_SEC:
        start = 0.0
        st.info(f"Song is {duration:.1f}s, so the whole song will be used.")
    else:
        start = st.slider(
            "Start of the 30-second segment (seconds)",
            min_value=0.0,
            max_value=float(duration - CLIP_SEC),
            value=0.0,
            step=1.0,
        )
    end = min(start + CLIP_SEC, duration)
    clip = y[int(start * sr): int(end * sr)]

    st.caption(f"Selected: {start:.0f}s → {end:.0f}s of {duration:.0f}s")
    st.pyplot(plot_waveform(y, sr, start, end))

    # preview of the selected clip only
    buf = io.BytesIO()
    sf.write(buf, clip, sr, format="WAV")
    st.audio(buf.getvalue(), format="audio/wav")

    if st.button("Predict genre"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, clip, sr)
            tmp_path = tmp.name

        try:
            with st.spinner("Analyzing the song..."):
                results = predict_genre(tmp_path)
                results_new_model = predict_genre_new(tmp_path)
                
                result_raw_svm = predict_svc_base_genre(tmp_path)
                result_svc_tuned = predict_svc_genre(tmp_path)

            st.divider()
           # HGB 
            top_genre, top_score = results[0]
            top_genre_new, top_score_new = results_new_model[0]
           # SVM 
            top_genre_svm, top_score_svm = result_svc_tuned[0]
            top_genre_raw_svm, top_score_raw_svm = result_raw_svm[0] 
            
            
            df = pd.DataFrame(results, columns=["genre", "score"]).set_index("genre")
            df_new = pd.DataFrame(results_new_model, columns=["genre", "score"]).set_index("genre")
           
            df_svm = pd.DataFrame(result_svc_tuned, columns=["genre", "score"]).set_index("genre")
            df_svm_raw = pd.DataFrame(result_raw_svm, columns=["genre", "score"]).set_index("genre")
             
            
            
            

            st.caption("Top 3, HGB")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Model 1, HGB")
                st.badge(top_genre)
                st.success(f"**{top_genre}** ({top_score:.0%})")
                st.bar_chart(df)

            with col2:
                st.markdown("#### Model 1 (tuned), HGB")
                st.badge(top_genre_new)   # was top_genre
                st.success(f"**{top_genre_new}** ({top_score_new:.0%})")
                st.bar_chart(df_new)
                
                
                
            st.caption("Top 3, SVC")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Model 2, SVM (classifier)")
                st.badge(top_genre_raw_svm)
                st.success(f"**{top_genre_raw_svm}** ({top_score_raw_svm:.0%})")
                st.bar_chart(df_svm)

            with col2:
                st.markdown("#### Model 2, Tuned SVM")
                st.badge(top_genre_svm)   # was top_genre
                st.success(f"**{top_genre_svm}** ({top_score_svm:.0%})")
                st.bar_chart(df_svm_raw)

        except Exception as e:
            st.exception(e)
        finally:
            os.remove(tmp_path)