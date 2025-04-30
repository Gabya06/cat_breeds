from PIL import Image
import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import CLIPModel, CLIPProcessor


model_name = "openai/clip-vit-base-patch32"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)


class ClipMatcher:
    @staticmethod
    def preprocess_text(text_list: List[str]) -> torch.Tensor:
        """
        Preprocess text data for CLIP model.

        Parameters:
        -----------
        text_list: List of text data to preprocess.

        Returns:
        ---------
        torch.Tensor: Normalized text embeddings (n_texts, hidden_dim).
        """
        with torch.no_grad():
            inputs = processor(
                text=text_list, return_tensors="pt", padding=True, truncation=True
            ).to(DEVICE)
            text_embeddings = clip_model.get_text_features(**inputs)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings.cpu().squeeze().tolist()

    @staticmethod
    def preprocess_image(image_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess an image for CLIP model.

        Parameters:
        image_path (str): Path to the image file.

        Returns:
        torch.Tensor: Preprocessed image embedding.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)  # type: ignore
                image_embeddings = clip_model.get_image_features(**inputs)  # type: ignore
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                return image_embeddings.cpu().squeeze().tolist()  # batched: List[List[float]]
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    @staticmethod
    def predict_breed(
        image_embedding: torch.Tensor,
        text_embeddings: torch.Tensor,
        breed_names: List[str],
        topk: int = 1,
        return_similarity: bool = False,
    ) -> List:
        """
        Predict the breed of a cat based on its image embedding and text embeddings.

        Parameters:
        -----------
        image_embedding: torch.Tensor
            Preprocessed image embedding.
        text_embeddings: torch.Tensor
            Preprocessed text embeddings.
        breed_names: List
            List of breed names.
        topk: int
            Number of results to return (top similar breeds). Default is 1.
        Returns:
        --------
        predicted_breed: List
            k [predicted_breed, confidence] if return_similarity = True.
            k [predicted_breed] if return_similarity = False
        """
        if image_embedding is None:
            print("Error: Image embedding is None.")
            return None

        predicted_breeds = []
        # calculate similarity between image and text embeddings
        similarity = F.cosine_similarity(image_embedding, text_embeddings)
        # find most similar item
        top_results = similarity.topk(k=topk)
        predicted_ix = top_results.indices
        confidence = top_results.values.tolist()

        for ix in predicted_ix:
            predicted_breeds.append(breed_names[ix])

        if return_similarity:
            return list(zip(predicted_breeds, confidence))
        else:
            return predicted_breeds
