from google import genai
from io import BytesIO
import pandas as pd
from PIL import Image
import tempfile
import streamlit as st

from app.logic import aggregate_predictions
from app.ui_helpers import get_cat_qa, get_cat_image_generator, load_breeds
from cat_breeds.infer import predict_breed_chroma, predict_breed_clip


# ---------- APP -----------

# Configuration and Intro
st.set_page_config(page_title="Cat Breed Predictor", layout="wide")
# st.title("🐾 Cat Breed Predictor 🐾")
st.title("🐱 Cat Intelligence Explorer 🐱")
st.markdown(
    """🐾  Explore cat breeds using image recognition, AI Q&A, and image generation.
    Powered by **CLIP**, **Gemini**, and **ChromaDB**.🐾
    """
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    # Session state defaults

    # Uploaded files
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = False

    uploaded_files = st.file_uploader(
        "Upload up to 3 cat images", accept_multiple_files=True, type=["jpg", "jpeg", "png"]
    )
    if uploaded_files:
        st.session_state.uploaded = True
        st.session_state.uploaded_files = uploaded_files

    if st.session_state.uploaded and st.button("🔁 Upload New Images"):
        st.session_state.uploaded = False
        st.session_state.uploaded_files = []
        # st.rerun()

    # Prediction Models to use: Chroma/CLIP
    selected_models = st.multiselect(
        "Select model(s) to use:",
        ["CLIP", "ChromaDB"],
        default=["CLIP", "ChromaDB"],
        help="Choose which Model to use: CLIP, Chroma or Both!",
    )
    # Top Predictions
    top_n_preds = st.slider(
        label="Number of Top Predictions 🐱",
        min_value=1,
        max_value=5,
        step=1,
        value=2,
        help="Select the Number of Predictions to Show for Your Fur Baby 🐱",
    )
    st.session_state.top_n_preds = top_n_preds

    model_options = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro-latest",
    ]
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_options[0]

    selected_gemini_model = st.selectbox(
        "Choose Gemini model",
        model_options,
        index=model_options.index(st.session_state.selected_model),
    )
    st.session_state.selected_model = selected_gemini_model


# Create tabs
tab1, tab2 = st.tabs(["📷 Image Breed Prediction", "❓ Q&A with Gemini"])
# Get Cat Breed Predictions
with tab1:
    st.markdown("### 📸 Compare Breed Predictions from Image")
    clip_pred_list, chroma_pred_list = [], []

    if st.session_state.uploaded:
        with st.spinner("Running Predictions ..."):
            file_names = [f.name for f in st.session_state.uploaded_files]
            # file_names = [i.name for i in uploaded_files]
            n_uploaded = min(len(file_names), 3)

            cols = st.columns(n_uploaded, gap="small")

            for col, f in zip(cols, st.session_state.uploaded_files[:3]):
                # Open image & resize
                image = Image.open(f).convert("RGB")
                resized_image = image.resize((800, 600))
                # Display image
                col.image(resized_image, caption=f.name)

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    image.save(tmp_file.name)

                if "CLIP" in selected_models:
                    clip_preds = predict_breed_clip(
                        tmp_file.name, topk=top_n_preds, return_similarity=True
                    )

                    clip_pred_list.append(clip_preds)

                if "ChromaDB" in selected_models:
                    chroma_preds = predict_breed_chroma(
                        tmp_file.name, topk=top_n_preds, return_similarity=True
                    )

                    chroma_pred_list.append(chroma_preds)

            # aggregate predictions
            final_preds = aggregate_predictions(clip_pred_list, chroma_pred_list)
            display_preds = final_preds[:top_n_preds]

            # store results in session
            st.session_state.top_breeds = [breed for breed, _ in display_preds]

            # Display final top predictions
            st.markdown("## 🏆 Top Overall Prediction")

            for i, (breed, score) in enumerate(display_preds):
                st.markdown(f"#### {breed} — Rank: {i+1} - Score: `{score:.3f}`")

            # Toggle for detailed view
            if st.toggle("🔍 Show Predictions by Model"):
                st.subheader("📊 Model-Specific Scores")
                col1, col2 = st.columns(2)

                clip_df = pd.DataFrame()
                chroma_df = pd.DataFrame()

                for i, (clip_img_preds, chroma_img_preds) in enumerate(
                    zip(clip_pred_list, chroma_pred_list)
                ):

                    # Convert to DataFrame for display
                    clip_temp = pd.DataFrame(
                        clip_img_preds,
                        columns=["Breed", "CLIP Score"],
                    )
                    clip_temp["Image"] = file_names[i]
                    clip_temp.sort_values(by=["CLIP Score"], ascending=False, inplace=True)
                    clip_temp["Rank"] = range(1, top_n_preds + 1)
                    clip_df = pd.concat([clip_df, clip_temp])

                    chroma_temp = pd.DataFrame(
                        chroma_img_preds, columns=["Breed", "ChromaDB Score"]
                    )

                    chroma_temp["Image"] = file_names[i]
                    chroma_temp.sort_values(by=["ChromaDB Score"], ascending=False, inplace=True)
                    chroma_temp["Rank"] = range(1, top_n_preds + 1)
                    chroma_df = pd.concat([chroma_df, chroma_temp])

                with col1:
                    st.dataframe(data=clip_df, hide_index=True, use_container_width=True)
                with col2:
                    st.dataframe(data=chroma_df, hide_index=True, use_container_width=True)

# Ask Cat Breed Questions
with tab2:
    st.markdown(" ### ✨ Choose a Breed to Explore")
    st.session_state.all_breeds = breeds = load_breeds()

    if "top_breeds" in st.session_state:
        # st.subheader("✨ Choose a Breed to Explore")
        breed_options = st.session_state.top_breeds + ["Other breed"]
        selected_breed = st.radio("Select a breed", breed_options)

        if selected_breed == "Other breed":
            breed_to_use = st.selectbox("Select a cat breed", options=breeds)

        else:
            breed_to_use = selected_breed
        default_question = f"Tell me something fun about the {breed_to_use} cat breed."
        user_question = st.text_input("Ask a question about this breed:", value=default_question)

        with st.spinner("Querying Gemini ..."):
            if "top_n_preds" in st.session_state:
                top_preds = st.session_state.top_n_preds
            else:
                top_preds = 2
            catQa = get_cat_qa()
            results = catQa.query(
                query=default_question,
                mode="text",
                n_results=top_preds,
                filters={"breed": breed_to_use},
            )

            if len(results["ids"][0]) == 0:
                st.markdown(f"No results found for {breed_to_use}. Try a different breed")
            try:
                answer = catQa.get_answer(model=selected_gemini_model)
                st.success("Answer:")
                st.markdown(answer)

                if st.toggle("Show Prompt"):
                    prompt = catQa.build_prompt()
                    st.html(f"Prompt: {prompt}")

                # create cat generator
                st.divider()
                if st.button(f"🔁 Generate Image of a {selected_breed}"):
                    cat_generator = get_cat_image_generator()
                    prompt = f"Generate an image of a {selected_breed} cat"
                    image_data = cat_generator.generated_cat_images(prompt=prompt)
                    if image_data:
                        try:
                            image = Image.open(BytesIO(image_data))
                            st.image(image=image)
                            # Prepare for download
                            buffer = BytesIO()
                            image.save(buffer, format="PNG")
                            buffer.seek(0)

                            st.download_button(
                                label="📥 Save Image",
                                data=buffer,
                                file_name=f"{selected_breed.lower().replace(' ', '_')}_cat.png",
                                mime="image/png",
                            )

                        except Exception as e:
                            print(f"Error Generating Image: {e}")
            except genai.errors.ServerError as e:
                st.error("🐱 Gemini is overloaded. Please try again in a few seconds.")

    else:
        st.warning("Run predictions in Breed Predictor tab first.")
