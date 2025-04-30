# 🐱 Cat Intelligence Project
A multimodal RAG (Retrieval-Augmented Generation) system powered by Gemini and ChromaDB that lets you explore cat breeds via text and image queries. Ask questions like "Which fluffy cats come from Russia?" and visualize the results with breed predictions, embeddings, and more!

## 🗂 Project Structure
``` md
cat_breeds/ 
└── src/ 
    ├── app.py # Streamlit entry point 
    ├── app/ # UI helpers and logic 
        └── ui_helpers.py 
        └── logic.py 
    ├── cat_breeds/ # Core modules (CLIP, ChromaDB, RAG) 
        └── cat_breed_qa.py 
        └── cat_data_processor.py 
        └── clip_matcher.py 
        └── create_chroma_embeddings.py
        └── infer_breeds.py
        ├── utils/ # Additional functionality
            └── embedding_functions.py # Embedding creation functions
            └── utils.py 
└── db/ # ChromaDB SQLite storage
└── notebooks/ # Jupyter demo notebooks
└── images/ # Images
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
from cat_breeds.cat_breed_qa import CatBreedQA
from cat_breeds.clip_predictor import predict_breed
```

* Ask a question:
``` py
query_breeds("Which cats have short legs and are affectionate?")
```

* Predict breed from an image:
``` py
predict_breed("cat_images/munchkin.jpg")
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


## 📸 Example Output
Coming soon: screenshots of search results and prediction visualizations.


🤝 Contributing
Feel free to open issues or PRs. Cat lovers and AI nerds welcome 🐾

