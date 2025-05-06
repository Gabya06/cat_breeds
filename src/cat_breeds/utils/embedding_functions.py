from typing import List, Optional

import torch
from cat_breeds.clip import ClipMatcher
from chromadb.utils.embedding_functions import EmbeddingFunction


class ClipTextEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function that converts text to CLIP text embeddings
    and returns a list of float lists for ChromaDB.
    """

    def __call__(self, texts: List[str]) -> Optional[List[List[float]]]:
        embeddings = ClipMatcher.preprocess_text(texts)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.tolist()
        return embeddings


class ClipImageEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function that converts image paths to CLIP image embeddings
    and returns a list of float lists for ChromaDB.
    """

    def __call__(self, image_paths: List[str]) -> Optional[List[List[float]]]:
        embeddings = [ClipMatcher.preprocess_image(p) for p in image_paths]
        return [e.tolist() if isinstance(e, torch.Tensor) else e for e in embeddings]
        # ClipMatcher.preprocess_image(p).tolist() for p in image_paths]
