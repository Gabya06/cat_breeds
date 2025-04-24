"""
Script to retrieve data and create ChromaDB embeddings. This should only be run once when
getting started or get new data and want to recreate embeddings.
"""

import os

from dotenv import load_dotenv
from google import genai
import pandas as pd
from tqdm import tqdm

from cat_breeds import ClipMatcher
from cat_breeds.utils.embedding_functions import (
    ClipTextEmbeddingFunction,
    ClipImageEmbeddingFunction,
)
from cat_breeds.utils import utils


load_dotenv()
# load Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def create_embeddings():
    data = pd.read_csv("/Users/gabyagrocostea/dev/cat_breeds/data/cat_data.csv")

    # Generate CLIP embeddings for images and breed descriptions
    embedding_documents = []
    image_embeddings = []
    text_embeddings = []
    metadatas = []

    # Loop through each row and generate embeddings for the image & cat_description
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        image_path = row["image_path"]
        text = row["cat_description"]

        image_embedding = ClipMatcher.preprocess_image(image_path=image_path)  # preprocess image
        text_embedding = ClipMatcher.preprocess_text(text_list=[text])  # preprocess text

        if image_embedding is not None:
            embedding_documents.append(
                f"Image: {row['breed']}"
            )  # Document description for image db
            image_embeddings.append(image_embedding)
            text_embeddings.append(text_embedding)

            # Add breed metadata
            breed_metadata = {
                "breed": row["breed"],
                "origin": row["origin"],
                "affection_level": row["affection_level"],
                "health_issues": row["health_issues"],
                "shedding_level": row["shedding_level"],
            }
            metadatas.append(breed_metadata)

    assert (
        len(embedding_documents) == len(image_embeddings) == len(metadatas) == len(text_embeddings)
    )

    # create text and image collections (Databases) using CLIP embeddings functions
    text_embedding_fn = ClipTextEmbeddingFunction()
    text_collection = utils.create_chromadb(
        db_name="cat_breed_texts", embedding_function=text_embedding_fn, delete_exisiting=False
    )

    image_embedding_fn = ClipImageEmbeddingFunction()
    image_collection = utils.create_chromadb(
        db_name="cat_breed_images", embedding_function=image_embedding_fn, delete_exisiting=False
    )

    # add metadata information to facilitate filtering
    metadatas = [
        {
            "breed": row["breed"],
            "origin": row["origin"],
            "affection_level": row["affection_level"],
            "health_issues": row["health_issues"],
            "shedding_level": row["shedding_level"],
        }
        for _, row in data.iterrows()
    ]

    # add documents and text embeddings to text collection
    text_collection.add(
        documents=[row["cat_description"] for _, row in data.iterrows()],
        ids=[str(idx) for idx in range(len(data))],
        embeddings=text_embeddings,
        metadatas=metadatas,
    )

    # add documents and image embeddings to image collection
    image_collection.add(
        documents=[f"Image: {row['breed']}" for _, row in data.iterrows()],
        ids=[str(idx) for idx in range(len(data))],
        embeddings=image_embeddings,
        metadatas=metadatas,
    )

    print(f"text count {text_collection.count()}")  # should return > 0
    print(f"image counts {image_collection.count()}")


if __name__ == "__main__":
    create_embeddings()
