# from glob import glob
from dotenv import load_dotenv
from glob import glob
from io import BytesIO
import os
import pandas as pd
from PIL import Image
import requests
import time
import warnings


load_dotenv()
CAT_API_KEY = os.getenv("CAT_API_KEY")

# Flag if the API is unavailable
if CAT_API_KEY:
    CAT_API_AVAILABLE = True
else:
    CAT_API_AVAILABLE = False
    warnings.warn(
        "CAT_API_KEY not found. Some functionality (like image downloading) may not work. "
        "Set the key in a `.env` file or via `os.environ['CAT_API_KEY'] = 'your_key_here'`"
    )

if not CAT_API_KEY and "COLAB_GPU" in os.environ:
    from getpass import getpass

    CAT_API_KEY = getpass("Enter your CAT_API_KEY: ")
    os.environ["CAT_API_KEY"] = CAT_API_KEY  # for downstream compatibility


# Base URL for CatAPI
BASE_URL = "https://api.thecatapi.com/v1/"


class CatDataProcessor:

    @classmethod
    def get_breeds(cls):
        """
        Get breed information from CatAPI
        """
        if not CAT_API_AVAILABLE:
            try:
                response = requests.get(BASE_URL + "breeds")
            except Exception as e:
                raise (f"Cat API is not available. Please enter CAT_API_KEY\n{e}")
        else:
            headers = {"x-api-key": CAT_API_KEY}
            response = requests.get(BASE_URL + "breeds", headers=headers)

        if response.status_code == 200:
            return response.json()  # Returns a list of breed dictionaries
        else:
            print("Error fetching breeds data.")
            return []

    @staticmethod
    def parse_life_span(life_span_str: str):
        """
        Parse lifespan information to retrieve min_life, max_life and avg_life

        Parameters:
        -----------
        life_span_str: str
            Represents lifespan, ie: '10 - 14'

        Returns:
        --------
        min_life, max_life, avg_life ie: 1 , 15 , 7
        """
        try:
            min_life, max_life = map(int, life_span_str.split(" - "))
            avg_life = round((min_life + max_life) / 2, 1)
            return min_life, max_life, avg_life
        except Exception:
            return None, None, None

    @staticmethod
    def download_images_per_breed(breed_id: str, breed_name: str, num_images: int = 10):
        """
        Download images per breed with incremental naming

        Parameters
        ----------
        breed_id : str
            ID of the breed (ie: 'abys' for Abyssinian)
        breed_name : str
            Name of the breed
        num_images : int
            Number of images to download
        """
        headers = {"x-api-key": CAT_API_KEY}
        url = "https://api.thecatapi.com/v1/images/search"
        params = {"breed_id": breed_id, "limit": num_images}

        # Create folder if it doesn't exist
        breed_folder = f"images/{breed_name}"
        os.makedirs(breed_folder, exist_ok=True)

        # Count how many images are already there
        existing_images = glob(os.path.join(breed_folder, f"{breed_name}_*"))
        next_index = len(existing_images) + 1

        # Get image URLs from Cat API
        response = requests.get(url, headers=headers, params=params)

        # Check for errors and add delay if rate limited
        if response.status_code == 429:  # Check if rate limited
            print("Rate limited! Waiting for 60 seconds...")
            time.sleep(60)
            response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(
                f"Error fetching images for {breed_name}: \
                    Status code {response.status_code}"
            )
            print(f"Response content: {response.content}")
            return

        images = response.json()
        for idx, img in enumerate(images):
            try:
                img_url = img["url"]
                img_ext = img_url.split(".")[-1].split("?")[0]  # get file extension
                out_path = os.path.join(breed_folder, f"{breed_name}_{next_index}.{img_ext}")

                img_response = requests.get(img_url)
                if img_response.status_code == 200:
                    image = Image.open(BytesIO(img_response.content))
                    image.save(out_path)
                    print(f"Saved: {out_path}")
                    next_index += 1
            except Exception as e:
                print(f"Error saving image for {breed_name}: {e}")
        # add time delay between requests to avoid hitting rate limits
        time.sleep(1)

    @staticmethod
    def process_data_for_rag(df: pd.DataFrame) -> pd.DataFrame:
        """ """
        affection_map = {3: "pretty affectionate", 4: "affectionate", 5: "very affectionate"}

        shedding_map = {
            5: "shed a lot more",
            4: "shed a lot",
            3: "shed an average amount",
            2: "shed",
            1: "do not shed a lot",
        }

        health_map = {
            1: "have no health issues",
            2: "have few health issues",
            3: "have some health issues",
            4: "can have health issues",
        }
        # Add text for affection_level, shedding_level and health_issues
        df.affection_level = df.affection_level.map(affection_map)
        df.shedding_level = df.shedding_level.map(shedding_map)
        df.health_issues = df.health_issues.map(health_map)

        df[["life_span_min", "life_span_max", "life_span_avg"]] = df["life_span"].apply(
            lambda x: pd.Series(CatDataProcessor.parse_life_span(x))
        )

        df["life_span_text"] = df["life_span_avg"].apply(
            lambda x: f"They typically live around {x} years." if pd.notnull(x) else ""
        )

        # assign text for each row - This will be used in RAG
        df = df.assign(
            cat_description=df.apply(
                lambda row: f"{row.description} Their temperament is described as "
                f"{row.temperament.strip()}"
                f"{'They are hypoallergenic' if row.hypoallergenic == 1 else ''}. "
                f"They are known to be {row.affection_level}, tend to {row.shedding_level} and "
                f"{row.health_issues}. "
                f"{row.life_span_text}",
                axis=1,
            )
        )
        df.rename(
            columns={
                "name": "breed",
            },
            inplace=True,
        )

        return df
