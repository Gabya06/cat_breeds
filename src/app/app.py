import streamlit as st
from PIL import Image
import tempfile

import matplotlib.pyplot as plt
import pandas as pd


from cat_breeds.infer_breed import predict_breed_chroma, predict_breed_clip


st.set_page_config(page_title="Cat Breed Predictor", layout="centered")
st.title("🐾 Cat Breed Predictor")
st.write("Compare breed predictions using CLIP vs ChromaDB embeddings.")

uploaded_file = st.file_uploader("Upload your cat image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Cat", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)

        st.subheader("🔍 Predictions")
        with st.spinner("Generating predictions..."):
            clip_preds = predict_breed_clip(tmp_file.name, topk=3, return_similarity=True)
            chroma_preds = predict_breed_chroma(tmp_file.name, topk=3, return_scores=True)

        # clip_breed, clip_sim = [f"{breed} {sim:.2f}" for breed, sim in clip_preds]
        # chroma_breed, chroma_dist = [
        #     f"{breed} {dist:.2f}" for breed, dist in chroma_preds  # type: ignore
        # ]

        # CLIP Predictions
        st.markdown("### 🤖 CLIP Model")
        for breed, sim in clip_preds:
            st.write(f"- {breed} (cosine similarity: **{sim:.2f}**) ")

        # Chroma Predictions
        st.markdown("### 🧠 ChromaDB Nearest Neighbor")
        for breed, dist in chroma_preds:  # type: ignore
            st.write(f"- {breed} (distance: **{dist:.2f}**) ")

        # Show bar comparison
        st.markdown("### 📊 Similarity vs Distance")

        clip_df = pd.DataFrame(
            {
                "breed": [b for b, _ in clip_preds],
                "score": [sim for _, sim in clip_preds],
                "method": "CLIP",
            }
        )

        chroma_df = pd.DataFrame(
            {
                "breed": [b for b, _ in chroma_preds],  # type: ignore
                # inverse distance to align with similarity
                "score": [1 - d for _, d in chroma_preds],
                "method": "Chroma",
            }
        )

        combined_df = pd.concat([clip_df, chroma_df])

        fig, ax = plt.subplots()
        for method, group in combined_df.groupby("method"):
            ax.barh(group["breed"], group["score"], label=method, alpha=0.6)
        ax.set_xlabel("Similarity / (1 - Distance)")
        ax.set_title("Model Confidence Comparison")
        ax.legend()
        st.pyplot(fig)

    st.success("Done!")
