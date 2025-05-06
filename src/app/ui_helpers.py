import asyncio
import os

from dotenv import load_dotenv
from google import genai
import pandas as pd
from pathlib import Path
import streamlit as st

from chromadb import PersistentClient
from cat_breeds.utils.embedding_functions import (
    ClipTextEmbeddingFunction,
    ClipImageEmbeddingFunction,
)
from cat_breeds.qa import CatBreedQA
from cat_breeds.cat_image_generator import CatImageGeenerator


def load_genai_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    return genai.Client(api_key=api_key)


@st.cache_resource
def load_collections():
    """
    Helper function to load existing text and image collections
    """
    # Set project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # goes from app.py → app → src → root
    DB_PATH = PROJECT_ROOT / "db"
    # Load collection from disk
    chroma_client = PersistentClient(path=str(DB_PATH))  # "db/")
    text_collection = chroma_client.get_collection(
        name="cat_breed_texts", embedding_function=ClipTextEmbeddingFunction()
    )
    image_collection = chroma_client.get_collection(
        name="cat_breed_images", embedding_function=ClipImageEmbeddingFunction()
    )
    return text_collection, image_collection


@st.cache_resource
def load_breeds():
    text_collection, _ = load_collections()
    metadatas = text_collection.get(include=["metadatas"])["metadatas"]
    breeds = [i.get("breed") for i in metadatas]  # type: ignore
    return breeds


# --- Set up CatBreedQA object ---
@st.cache_resource
def get_cat_qa():
    if "genai_client" not in st.session_state:
        st.session_state.genai_client = load_genai_client()
    client = st.session_state.genai_client

    # client = load_genai_client()
    text_collection, image_collection = load_collections()
    return CatBreedQA(
        text_db=text_collection, image_db=image_collection, client=client, data=pd.DataFrame()
    )


# --- Set up CatImageGenerator object
@st.cache_resource
def get_cat_image_generator():
    if "genai_client" not in st.session_state:
        st.session_state.genai_client = load_genai_client()
    client = st.session_state.genai_client
    cat_image_generator = CatImageGeenerator(client=client)
    return cat_image_generator
