import os
import io
import zipfile
import requests
import pandas as pd
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextDataLoader:
    """
    Handles loading, cleaning, and splitting of the text classification dataset.
    Prioritizes 'Sport' and 'Politics' categories from the BBC News dataset.
    """

    DATA_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    DATA_DIR = "data"
    DATASET_FOLDER_NAME = "bbc"

    def __init__(self, target_categories: List[str] = ['sport', 'politics']):
        """
        Initializes the data loader.

        Args:
            target_categories (List[str]): List of categories to include. Defaults to ['sport', 'politics'].
        """
        self.target_categories = target_categories
        self.data_path = os.path.join(self.DATA_DIR, self.DATASET_FOLDER_NAME)

    def download_data(self):
        """
        Downloads and extracts the BBC News dataset if it doesn't already exist.
        """
        if os.path.exists(self.data_path):
            logging.info(f"Dataset already exists at {self.data_path}. Skipping download.")
            return

        logging.info(f"Downloading dataset from {self.DATA_URL}...")
        try:
            response = requests.get(self.DATA_URL)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(self.DATA_DIR)
                
            logging.info("Dataset downloaded and extracted successfully.")
        except Exception as e:
            logging.error(f"Failed to download dataset: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from disk, filtering for the target categories.

        Returns:
            pd.DataFrame: A DataFrame containing 'text' and 'category' columns.
        """
        self.download_data()
        
        data = []
        
        logging.info(f"Loading data for categories: {self.target_categories}")
        
        for category in self.target_categories:
            category_path = os.path.join(self.data_path, category)
            
            if not os.path.exists(category_path):
                logging.warning(f"Category folder '{category}' not found at {category_path}. Skipping.")
                continue
                
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                
                # Check if it's a text file
                if os.path.isfile(file_path):
                    try:
                        # Attempt to read with 'latin-1' encoding which is common for this dataset
                        # if utf-8 fails, or just default to latin-1 as safe bet for BBC dataset
                        with open(file_path, 'r', encoding='latin-1') as f:
                            text = f.read()
                            data.append({'text': text, 'category': category})
                    except Exception as e:
                        logging.error(f"Error reading file {file_path}: {e}")

        df = pd.DataFrame(data)
        logging.info(f"Loaded {len(df)} documents.")
        return df

if __name__ == "__main__":
    # Test the loader
    loader = TextDataLoader()
    df = loader.load_data()
    print(df.head())
    print(df['category'].value_counts())
