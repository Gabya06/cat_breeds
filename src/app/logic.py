from collections import defaultdict
from typing import List, Tuple

import pandas as pd


def combine_results(
    clip_res: List[Tuple[str, float]], chroma_res: List[Tuple[str, float]]
) -> pd.DataFrame:
    """
    Combine CLIP and ChromaDB Results into one DataFrame

    Parameters:
    -----------
    clip_res: List
    chroma_res: List

    Return:
    -------
    combined_df: pd.DataFrame
        Contains breed and scores for each method

    """
    clip_df = pd.DataFrame(
        {
            "breed": [b for b, _ in clip_res],
            "score": [sim for _, sim in clip_res],
            "method": "CLIP",
        }
    )

    chroma_df = pd.DataFrame(
        {
            "breed": [b for b, _ in chroma_res],  # type: ignore
            # inverse distance to align with similarity
            "score": [1 - d for _, d in chroma_res],
            "method": "Chroma",
        }
    )

    combined_df = pd.concat([clip_df, chroma_df])
    return combined_df


def normalize(preds):
    """Normalize scores within a single image's predictions."""
    normalized = []
    for image_preds in preds:
        total = sum(score for _, score in image_preds)
        if total == 0:  # avoid divide-by-zero
            normalized.append([(breed, 0.0) for breed, _ in image_preds])
        else:
            normalized.append([(breed, score / total) for breed, score in image_preds])
    return normalized


def aggregate_predictions(clip_preds, chroma_preds, model_weights=None):
    """
    Aggregate CLIP and Chroma predictions:
        Normalize scores for each result (so they add up to 100%)
        Add scores for each predicted breed and sort them based on top scores
        Note: ChromaDB uses a psuedo-similarity score (1-distance)
    """
    breed_scores = defaultdict(float)

    # Optional: allow weighting models differently
    model_weights = model_weights or {"CLIP": 1.0, "ChromaDB": 1.0}

    # Normalize scores
    clip_preds_norm = normalize(clip_preds)
    chroma_preds_norm = normalize(chroma_preds)

    # Process CLIP predictions
    for image_preds in clip_preds_norm:
        for breed, score in image_preds:
            breed_scores[breed] += model_weights["CLIP"] * score

    # Process ChromaDB predictions
    for image_preds in chroma_preds_norm:
        for breed, score in image_preds:
            breed_scores[breed] += model_weights["ChromaDB"] * score

    # Sort by total score
    sorted_breeds = sorted(breed_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_breeds


# Confidence mapping
def get_confidence_label(score):
    if score >= 0.35:
        return "High Match"
    elif score >= 0.25:
        return "Moderate Match"
    else:
        return "Close Match"
