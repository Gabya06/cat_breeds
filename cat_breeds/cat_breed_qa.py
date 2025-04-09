import matplotlib.pyplot as plt
from PIL import Image
import textwrap


class CatBreedQA:
    def __init__(self, db, data, client):
        self.db = db
        self.data = data
        self.client = client

    def query(self, query: str, n_results: int = 3, filters: dict = None):
        self.query_text = query
        self.results = self.db.query(query_texts=[query], n_results=n_results, where=filters)
        return self.results

    def display(self, filter_desc: str = ""):
        """
        Display results by plotting breed images with scores
        """
        n_results = len(self.results["ids"][0])
        fig, axes = plt.subplots(ncols=1, nrows=n_results, figsize=(15, 5))

        if n_results == 1:
            axes = [axes]

        for ax, id, doc, score in zip(
            axes, self.results["ids"][0], self.results["documents"][0], self.results["distances"][0]
        ):

            idx = int(id)
            breed_name = self.data.iloc[idx].breed
            image_path = self.data.iloc[idx].image_path
            img = Image.open(image_path)

            ax.imshow(img)
            ax.set_title(f"{breed_name}\nScore: {round(float(score), 2)}", fontsize=10)
            ax.axis("off")
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

        plt.suptitle(f"{self.query_text}\n ({filter_desc})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def build_prompt(self):
        """
        Build prompt for LLM
        """
        # Combine docs into a single context string
        context = "\n\n".join(self.results["documents"][0])
        # Prompt the LLM with retrieved context

        prompt = textwrap.dedent(
            f"""
        You are a helpful and informative expert bot on cat breeds that answers questions
        using text from the reference passage included below. Be sure to respond in a complete
        sentence, being comprehensive, including all relevant background information.
        However, you are talking to cat lovers that might have differing preferences, so be sure
        to break down concepts and strike a friendly and conversational tone.
        While distinguishing different cat breeds by coat, body type and location of origin can be
        difficult, if the information is not relevant you may ignore it.

        Reference passage:
        {context}

        Question: {self.query_text}
        """
        )

        return prompt

    def get_answer(self):
        prompt = self.build_prompt()
        response = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text
