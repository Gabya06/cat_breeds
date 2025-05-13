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
import pandas as pd
import os

from cat_breeds.qa import CatBreedQA
from cat_breeds.infer import predict_breed_clip
from cat_breeds.utils import load_embeddings

# load Gemini API Key
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
data = pd.read_csv("data/cat_breeds.cv", index_col=0)
```

### Load Embeddings & Set up CatQA
``` py
text_collection, image_collection = load_embeddings()

catQa = CatBreedQA(text_db=text_collection, image_db=image_collection, client=client, data=data)
```

## Ask a Question!
``` py
question = "Which cats have short legs and are affectionate?"
text_example = catQa.query(query=question, mode="text", n_results=1)
text_results = catQa.get_answer()
```

### Results:
``` html
 Okay, I'd be happy to tell you about cats with short legs and affectionate personalities!

Based on the information I have, the **Munchkin** cat breed fits that description. Here's a bit more about them:

*   Munchkins are known to be **very affectionate** and have a playful, intelligent personality.
*   Originating in the United States, they're also known to **shed an average amount**.
*   It's worth noting that Munchkins **have some health issues**, so it's something to keep in mind.

```

### Visualize Results:
``` py
catQA.display()
```

![Short Legged Munchkin](/images/Munchkin/Munchkin_result.jpg)

## Predict Breed from an Image:
``` py
# Replace with your image path
breed = predict_clip_breed(image_path = "images/my_pets/bellini_1.png", topk=1, 
                                    return_similarity=False)
print(f"Your Predicted Cat Breed is: {breed}")
```

### Results:
Your Predicted Cat Breed is: **Persian**

----------------------------

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

