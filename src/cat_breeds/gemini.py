import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings


# Load API Key
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
# client = genai.Client(api_key=api_key)

BATCH_SIZE = 100


def get_gemini_api_key():
    # Try loading from `.env` if running locally
    if Path(".env").exists():
        load_dotenv()

    # Try environment variable (Colab, Kaggle)
    api_key = os.getenv("GEMINI_API_KEY")

    # Special case: Kaggle Secrets (Kaggle secrets tab must be manually added)
    if not api_key and Path("/kaggle").exists():
        kaggle_secrets_path = "/kaggle/working/secrets/gemini_api_key.txt"
        if Path(kaggle_secrets_path).exists():
            with open(kaggle_secrets_path) as f:
                api_key = f.read().strip()

    if not api_key:
        raise ValueError(
            "❌ GEMINI_API_KEY not found. Please set it in your `.env` file, "
            "environment variables, or upload it as a Kaggle secret."
        )

    return api_key


def is_retriable(e):
    """Define a retry predicate for Gemini API errors."""
    return isinstance(e, genai.errors.APIError) and e.code in {429, 503}


class EmbeddingModel(EmbeddingFunction):
    """
    Wrapper for Gemini embedding functions.
    """

    def __init__(self, document_mode=True):
        self.document_mode = document_mode

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generates embeddings for given input documents or queries in batches.
        """
        api_key = get_gemini_api_key()
        client = genai.Client(api_key=api_key)

        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        embeddings = []

        # process input in batches
        for i in range(0, len(input), BATCH_SIZE):
            batch = input[i : i + BATCH_SIZE]

            response = client.models.embed_content(
                model="models/text-embedding-004",
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings.extend([e.values for e in response.embeddings])
        return embeddings
