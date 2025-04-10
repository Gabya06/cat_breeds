import matplotlib.pyplot as plt
from PIL import Image
import textwrap
from typing import Dict, Optional
from cat_breeds.clip_matcher import ClipMatcher


class CatBreedQA:
    def __init__(self, text_db, image_db, data, client):
        self.text_db = text_db
        self.image_db = image_db
        self.data = data
        self.client = client

    def query(
        self,
        query: str,
        mode: str = "text",
        n_results: int = 3,
        filters: Optional[Dict] = None,
        image_path=None,
    ):
        """
        Query ChromaDB database and obtain top n results

        This functions can perfom:

        Text-to-Image Search:
            Find closest image vectors
        Image-to-Image Search
            Find similar cat images

        Parameters:
        ----------
        query: str
            Query, ie: "What is the most affectionate cat?"
        mode: str
            Can be: `text`, `image` or `image_to_text`. Default is `text`
            Describes the type of query.
                If using text to text query, use `text`.
                If doing image-to-image then use `image`.
                If using text to image query, use `text_to_image`.
        n_results: int
            Top n results to return
        filters: dict
            Dict with metadata to filter. ie: {'affection_level':'very affectionate'}
        query_embeddings: Optional[List]
            List of image or text embeddings to use when generating results
        """
        self.query_text = query
        self.mode = mode

        if mode == "text":
            results = self.text_db.query(query_texts=[query], n_results=n_results, where=filters)
        elif mode == "image" and image_path:
            query_embed = ClipMatcher.preprocess_image(image_path=image_path)
            results = self.image_db.query(query_embeddings=[query_embed], n_results=n_results)
        elif mode == "text_to_image":
            query_embed = ClipMatcher.preprocess_text([query])
            results = self.image_db.query(query_embeddings=[query_embed], n_results=n_results)
        else:
            raise ValueError("Invalid mode or missing image_path for image query")
        self.results = results
        return self.results

    def display(self, filter_desc: str = ""):
        """
        Display results by plotting breed images with scores
        """
        n_results = len(self.results["ids"][0])
        fig_height = max(3 * n_results, 5)  # Dynamic height: 3 inches per image
        fig, axes = plt.subplots(
            ncols=1, nrows=n_results, figsize=(15, fig_height), facecolor="white"
        )

        if n_results == 1:
            axes = [axes]

        for ax, id, doc, score in zip(
            axes, self.results["ids"][0], self.results["documents"][0], self.results["distances"][0]
        ):

            idx = int(id)
            breed_name = self.data.iloc[idx].breed
            image_path = self.data.iloc[idx].image_path

            try:
                img = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Could not open image at {image_path}: {e}")
                continue

            ax.imshow(img)
            ax.set_title(f"{breed_name}\nScore: {round(float(score), 2)}", fontsize=10)
            ax.axis("off")
            ax.set_facecolor("white")
            ax.text(
                0.5,
                -0.2,
                doc[:70] + "...",
                transform=ax.transAxes,
                ha="center",
                va="top",
                wrap=True,
                fontsize=8,
                color="dimgray",
            )
        s = f"{self.query_text}"
        if filter_desc:
            s += f"\n({filter_desc})"
        plt.suptitle(s, fontsize=14, fontweight="bold")
        # plt.suptitle(f"{self.query_text}\n ({filter_desc})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.9)
        plt.show()

    def build_prompt(self):
        """
        Build prompt for LLM
        """

        """
        Build prompt for LLM. Handles both text and image-based contexts.
        """
        # Combine docs into a single context string
        docs = self.results["documents"][0]
        has_descriptive_text = any(len(doc.strip().split()) > 3 for doc in docs)

        if has_descriptive_text:
            # Text-based retrieval
            context = "\n\n".join(docs)
            reference_intro = "Reference passage:"
        else:
            # Image-based retrieval with breed names only
            breed_names = [self.data.iloc[int(idx)].breed for idx in self.results["ids"][0]]
            context = "\n".join([f"Image: {name}" for name in breed_names])
            reference_intro = "The Reference includes images"

        prompt = textwrap.dedent(
            f"""
            You are a helpful and informative expert bot on cat breeds that answers questions
            using text from the reference passage{'s' if has_descriptive_text else ' or images'} 
            included below.
            If the reference passage include images and if you do not have access to these, feel 
            free to ignore the images. Mention that you are deriving your answer from your own 
            knowledge base, since you are an expert on cat breeds. 
            Be sure to respond in a complete sentence, being comprehensive, including all relevant 
            background information that you have been given.
            However, you are talking to cat lovers that might have differing preferences, so be sure
            to break down concepts and strike a friendly and conversational tone.
            While distinguishing different cat breeds by coat, body type and location of origin can 
            be difficult, if the information is not relevant you may ignore it.

            {reference_intro}
            {context}

            Question: {self.query_text}
            """
        )

        return prompt

    def get_answer(self):
        prompt = self.build_prompt()
        response = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text
