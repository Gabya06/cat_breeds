from chromadb import PersistentClient
from cat_breeds.utils.embedding_functions import (
    ClipTextEmbeddingFunction,
    ClipImageEmbeddingFunction,
)

from pathlib import Path


def create_chromadb(db_name: str, embedding_function, delete_exisiting: bool = False):
    """
    Create ChromaDB collection with given name and embedding function.

    Parameters
    ----------
    db_name : str
        Name of the ChromaDB collection.
    embedding_function : EmbeddingFunction
        Function to generate embeddings.
    delete_exisiting: Bool
        Whether to delete exisiting chromaDB before adding documents. Defaults to True
    """

    # create db and include metadata
    chroma_client = PersistentClient(path="db/")  # Persist to disk

    # delete existing collection
    if delete_exisiting:
        try:
            chroma_client.delete_collection(name=db_name)
        except Exception:
            pass
    # db = chroma_client.create_collection(name=db_name, embedding_function=embedding_function)
    db = chroma_client.get_or_create_collection(name=db_name, embedding_function=embedding_function)

    return db


def load_embeddings():
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
