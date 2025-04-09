🐾 Project Overview: Zero-Shot Cat Breed Classification with CLIP

This project explores the use of multimodal machine learning for classifying cat breeds in a zero-shot setting using CLIP (Contrastive Language–Image Pretraining). Instead of training a model on labeled cat breed images, we leverage CLIP's ability to align images and natural language descriptions in a shared embedding space.

We use Wikipedia text descriptions of cat breeds as the “reference knowledge” and compare them against cat images, either generated or retrieved, to determine the most likely breed—without any supervised training.

In addition, we explore retrieval-augmented generation (RAG) using ChromaDB to power natural language question-answering about cat breeds. This allows users to query breed-specific traits like “What does a Ragdoll cat look like?” and get a rich, grounded response, combining semantic retrieval with an LLM.

This project showcases:
* Zero-shot inference using CLIP on a custom cat breed dataset
* The use of embeddings for visual and textual similarity
* Integration of a vector database (ChromaDB) for RAG-style QA
* Optional use of Gemini for image generation and LLM answers