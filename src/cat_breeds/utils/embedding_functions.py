from typing import List

from cat_breeds.clip_matcher import ClipMatcher
from chromadb.utils.embedding_functions import EmbeddingFunction


class ClipTextEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return ClipMatcher.preprocess_text(texts)


class ClipImageEmbeddingFunction(EmbeddingFunction):
    def __call__(self, image_paths: List[str]) -> List[List[float]]:
        return [ClipMatcher.preprocess_image(p) for p in image_paths]
