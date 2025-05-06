from pathlib import Path
from chromadb import PersistentClient
import torch

from cat_breeds.clip import ClipMatcher

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # [2]  # goes from app.py → app → src → root
DB_PATH = PROJECT_ROOT / "db"

# Load collection from disk
chroma_client = PersistentClient(path=str(DB_PATH))
text_collection = chroma_client.get_collection(name="cat_breed_texts")
image_collection = chroma_client.get_collection(name="cat_breed_images")
metadatas = text_collection.get(include=["metadatas"])["metadatas"]
breeds = [i.get("breed") for i in metadatas]  # type: ignore


def predict_breed_clip(image_path: str, topk: int = 1, return_similarity: bool = True):
    """
    Predict cat breed based on CLIP topk similairities:
    - Embed image
    - Retrieve text embeddings from ChromaDB
    - Use ClipMatcher to calculate cosine similarity between image embedding and text embeddings

    Parameters:
    -----------
    image_path: str
        Path to image
    topk: int
        Number of results to return (top similar breeds). Default is 1.
    return_similarity: bool
        Defaults to True. Return confidence/similarity

    Returns:
    -------
    predicted_breed: List
        If return_similarity = True: return [predicted_breeds, confidence]
        If return_similarity = True: return [predicted_breeds]
    """
    image_embedding = ClipMatcher.preprocess_image(image_path=image_path)
    if not isinstance(image_embedding, torch.Tensor):
        image_tensor = torch.tensor(image_embedding)
    else:
        image_tensor = image_embedding.clone().detach()
    text_tensor = torch.tensor(
        text_collection.get(include=["embeddings", "metadatas"])["embeddings"]
    )

    predicted_breed = ClipMatcher.predict_breed(
        image_embedding=image_tensor,
        text_embeddings=text_tensor,
        breed_names=breeds,  # type: ignore
        topk=topk,
        return_similarity=return_similarity,
    )
    return predicted_breed


def predict_breed_chroma(image_path: str, topk: int = 1, return_similarity: bool = True):
    """
    ChromaDB distances - lower = better. Similarity is calculated by 1 - distance
    because in chromaDB lower distances are better, but we want higher similarity.
    So 1 - distance is a pseudo similarity to match intuition of CLIP
    """
    # preprocess image
    image_embedding = ClipMatcher.preprocess_image(image_path=image_path)
    # update to list of it's a tensor
    if isinstance(image_embedding, torch.Tensor):
        image_embedding = image_embedding.tolist()

    results = image_collection.query(
        query_embeddings=[image_embedding], n_results=topk  # type: ignore
    )  # type: ignore
    if return_similarity:
        return list(
            zip(
                [i["breed"] for i in results["metadatas"][0]],  # type: ignore
                [1 - i for i in results["distances"][0]],  # type: ignore
            )
        )
    else:
        return results["metadatas"][0][0]["breed"]  # type: ignore
