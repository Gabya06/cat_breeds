from google.genai import types


class CatImageGeenerator:
    def __init__(self, client):
        self.client = client

    def generated_cat_images(self, prompt: str):
        """
        Generate cat images using `gemini-2.0-flash-exp-image-generation`

        Parameters:
        ----------
        prompt: str
            Input prompt to generate image

        Return:
        -------
        image data

        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=[prompt],
                config=types.GenerateContentConfig(response_modalities=["Image", "Text"]),
            )
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        # image = Image.open(BytesIO(part.inline_data.data))
                        return part.inline_data.data
            print(f"Warning: No PNG image found in the response for prompt {prompt}")
            return None
        except Exception as e:
            print(f"Error generating image for prompt {prompt} - {e}")
            return None
