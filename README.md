# 🐱 Cat Intelligence Project
A multimodal RAG (Retrieval-Augmented Generation) system powered by Gemini and ChromaDB that lets you explore cat breeds via text and image queries. Ask questions like "Which fluffy cats come from Russia?" and visualize the results with breed predictions, embeddings, and more!

## 🗂 Project Structure
``` md
cat_breeds/ 
└── notebooks/                          -- Jupyter demo notebooks
└── src/ 
    ├── app.py                          -- Streamlit entry point 
    ├── app/                            -- UI helpers and logic 
        └── ui_helpers.py 
        └── logic.py 
    ├── cat_breeds/                     -- Python package (core logic) 
        └── qa.py                       -- Search ChromaDB, return results, build prompt
        └── data_processing.py          -- Preprocess cat API data 
        └── clip.py                     -- Preprocess text & images with CLIP
        └── embed.py                    -- One time run to create/update embeddings
        └── infer.py                    -- Predict cat breeds using CLIP & Chroma
        ├── utils/                      
            └── embedding_functions.py  -- Create CLIP embedding functions for images & text
            └── utils.py                -- Create Chroma.db
└── db/                                 -- ChromaDB SQLite storage

└── images/                             -- Images
└── Dockerfile                          -- DockerFile
└── .dockerignore                       -- Docker files to ignore
└── pyproject.toml                      -- Project Dependencies
```

## 📦 Features
### ✅ CAT API Data Retrieval
Fetch breed metadata and images from TheCatAPI.

### 🧠 RAG-style Q&A System
Query breed data using ChromaDB with filters (origin, traits, etc.) and pass retrieved docs to a Gemini LLM for enriched answers.

### 🐾 Multimodal Search
Match cat images to breed descriptions using CLIP and cosine similarity.

### 📊 Visualization
Display cat breed results using matplotlib and seaborn.

## 🚀 Installation (Colab or Local)
Clone the repo:

``` bash

git clone https://github.com/gabya06/cat_breeds.git
cd cat_breeds
pip install -e .
``` 

Install as a package (in your virtual environment or Colab):

``` bash
pip install -e .
``` 

## 🐾 Run the Streamlit App

Start the interactive UI for breed prediction and Q&A:

```bash
cd src
streamlit run app.py
```

## 🧪 Usage
``` py
from cat_breeds.qa import CatBreedQA
from cat_breeds.infer import predict_breed_clip
```

* Ask a question:
``` py
query_breeds("Which cats have short legs and are affectionate?")
```

* Predict breed from an image:
``` py
breed, score = predict_clip_breed(image_path = "cat_images/munchkin.jpg", topk=1, return_similarity=True)
```

## 🛠 Dependencies
Key libraries:

* chromadb
* openai, google-genai
* transformers, torch
* matplotlib, seaborn
* Pillow, requests

## 💡 Inspiration
Combines RAG, Gemini embeddings, ChromaDB filtering, and CLIP for an exploratory and educational multimodal AI demo around cats.


## 📸 Google Cloud
The app is now viewable on Google Cloud:
[🐱 Link to App](https://cat-breeds-app-624036724229.us-east1.run.app/)


### 🤝 Contributing
Feel free to open issues or PRs. Cat lovers and AI nerds welcome 🐾

