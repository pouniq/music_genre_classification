import os
import tempfile

import pandas as pd
import streamlit as st

from predict import predict_genre

st.set_page_config(
    page_title="Song Genre Classification",
    page_icon="🎵",
    layout="centered",
)

st.title("Song Genre Classification")
st.write("Upload a song & click **Predict genre**")
st.subheader("Audio file")

uploaded = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
)

if uploaded is not None:
    st.audio(uploaded)

    if st.button("Predict genre"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("Analyzing the song..."):
                results = predict_genre(tmp_path)  # [(genre, score), ...] top 3

            st.divider()
            top_genre, top_score = results[0]
            st.success(f"Predicted genre: **{top_genre}** ({top_score:.0%})")

            st.caption("Top 3")
            df = pd.DataFrame(results, columns=["genre", "score"]).set_index("genre")
            st.bar_chart(df)
        except Exception as e:
            st.exception(e)  # shows the full traceback, including the failing line
        finally:
            os.remove(tmp_path)