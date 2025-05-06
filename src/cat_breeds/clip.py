from PIL import Image
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import CLIPModel, CLIPProcessor


model_name = "openai/clip-vit-base-patch32"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)


class ClipMatcher:
    @staticmethod
    def preprocess_text(text_list: List[str]) -> Optional[torch.Tensor]:
        """
        Convert list of text strings to CLIP embeddings.
        Returns a tensor of shape (batch_size, embedding_dim).

        Parameters:
        -----------
        text_list: List of text data to preprocess.

        Returns:
        ---------
        torch.Tensor: Normalized text embeddings (n_texts, hidden_dim).
        """
        with torch.no_grad():
            inputs = clip_processor(
                text=text_list, return_tensors="pt", padding=True, truncation=True
            ).to(  # type: ignore
                DEVICE
            )
            text_embeddings = clip_model.get_text_features(**inputs)  # type: ignore
            text_embeddings = text_embeddings / text_embeddings.norm(
                dim=-1, keepdim=True
            )  # Normalize

        # in case empty embeddings
        if text_embeddings is None or text_embeddings.shape[0] == 0:
            print("Error: embeddings are empty")
            return None

        text_embeddings = text_embeddings.cpu()
        if text_embeddings.shape[0] == 1:
            text_embeddings = text_embeddings.squeeze(0)
        return text_embeddings
        # return text_embeddings.cpu().squeeze().tolist()

    @staticmethod
    def preprocess_image(image_path: str) -> Optional[torch.Tensor]:
        """
        Preprocess an image for CLIP model.

        Parameters:
        -----------
        image_path (str): Path to the image file.

        Returns:
        --------
        torch.Tensor: Preprocessed image embedding.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = clip_processor(images=image, return_tensors="pt")  # type: ignore
                inputs = inputs.to(DEVICE)
                image_embeddings = clip_model.get_image_features(**inputs)  # type: ignore
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                # return image_embeddings.cpu().squeeze().tolist()  # batched: List[List[float]]
                # make sure has correct dimensionality
                image_embeddings = image_embeddings.view(1, -1)
                return image_embeddings.cpu().squeeze(0)  # batched: List[List[float]]
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

    @staticmethod
    def predict_breed(
        image_embedding: torch.Tensor,
        text_embeddings: Union[List[torch.Tensor], torch.Tensor],
        breed_names: List[str],
        topk: int = 1,
        return_similarity: bool = False,
    ) -> Optional[Union[List[str], List[Tuple[str, float]]]]:
        """
        Predict the breed of a cat based on its image embedding and text embeddings.

        Parameters:
        -----------
        image_embedding: torch.Tensor
            Preprocessed image embedding (shape [512] or [1, 512]).
        text_embeddings: List[torch.Tensor] or torch.Tensor
            Preprocessed text embeddings (shape [N, 512]).
        breed_names: List
            List of breed names matching text embeddings.
        topk: int
            Number of top predictions. Default is 1.
        return_similarity: bool
            Whether to return similarity scores with breed names,

        Returns:
        --------
        predicted_breed: List[str] or List[Tuple[str, float]]
            List of predicted breed(s), optionally with confidence scores:
            - If return_similarity = True: [(breed1, score1), (breed2, score2), ...]
            - If return_similarity = False: [breed1, breed2, ...]
        """
        if image_embedding is None:
            print("Error: Image embedding is None.")
            return None
        # Convert list of embeddings to stack [N, 521]
        if isinstance(text_embeddings, list):
            text_embeddings = torch.stack(text_embeddings)
        # Enforce correct dimensions
        if image_embedding.ndim == 1:
            image_embedding = image_embedding.view(1, -1)
        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings.view(1, -1)

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
