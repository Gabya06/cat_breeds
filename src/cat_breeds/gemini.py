import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings


# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
client = genai.Client(api_key=api_key)

BATCH_SIZE = 100


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
