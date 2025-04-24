"""
Script to test out CLIP and Chroma breed inference for a few images
"""

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from cat_breeds.infer_breed import predict_breed_clip, predict_breed_chroma


df = pd.DataFrame()
test_images = [
    "images/my_pets/bellini_1.png",
    "images/my_pets/leo_1.png",
    "images/York Chocolate/York Chocolate_1.jpg",
    "images/Turkish Van/Turkish Van_1.jpg",
    "images/my_pets/bellini_2.png",
    "images/my_pets/motzy_1.png",
    "images/my_pets/cookie.png",
    "images/my_pets/mavie_1.png",
    "images/my_pets/bebe_1.png",
    "images/my_pets/bebe_2.png",
    "images/my_pets/bebe_3.png",
]


clip_preds = []
chroma_preds = []

for image_path in test_images:

    predicted_breed_clip = predict_breed_clip(image_path=image_path, topk=2, return_similarity=True)
    predicted_breed_chroma = predict_breed_chroma(image_path=image_path, topk=2, return_scores=True)

    clip_breed, clip_sim = [f"{breed} {sim:.2f}" for breed, sim in predicted_breed_clip]
    chroma_breed, chroma_dist = [
        f"{breed} {dist:.2f}" for breed, dist in predicted_breed_chroma  # type: ignore
    ]
    print(f"Image {image_path}")
    print(f"🟡 Model 1 (CLIP) results: {clip_breed}\n({clip_sim})")
    print(f"🟡 Model 2 (Chroma) results: {chroma_breed}\n({chroma_dist})")

    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"CLIP:{clip_breed}\nChroma: {chroma_breed}")
    plt.show()
    clip_preds.append(predicted_breed_clip)
    chroma_preds.append(predicted_breed_chroma)
