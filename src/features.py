from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from typing import Any, Tuple

class FeatureExtractor(ABC):
    """
    Abstract base class for feature extraction strategies.
    This class defines the interface for transforming text data into numerical features.
    """
    
    @abstractmethod
    def fit_transform(self, texts: pd.Series) -> Tuple[Any, Any]:
        """
        Fits the extractor to the data and transforms the texts into features.

        Args:
            texts (pd.Series): The text data to transform (e.g., training data).

        Returns:
            Tuple[Any, Any]: A tuple containing the transformed features (usually a sparse matrix)
                             and the fitted vectorizer object.
        """
        pass

    @abstractmethod
    def transform(self, texts: pd.Series) -> Any:
        """
        Transforms new texts using the already fitted extractor.
        
        Args:
            texts (pd.Series): New text data to transform (e.g., test data).
            
        Returns:
            Any: Transformed features (usually a sparse matrix).
        """
        pass

class BoWExtractor(FeatureExtractor):
    """
    Bag of Words Feature Extractor.
    Represents text as a count of word occurrences.
    """
    def __init__(self):
        # Initialize CountVectorizer with English stop words removal
        self.vectorizer = CountVectorizer(stop_words='english')

    def fit_transform(self, texts: pd.Series) -> Tuple[Any, Any]:
        """
        Fits the CountVectorizer and returns the BoW matrix.
        """
        features = self.vectorizer.fit_transform(texts)
        return features, self.vectorizer

    def transform(self, texts: pd.Series) -> Any:
        """
        Transforms texts using the fitted CountVectorizer.
        """
        return self.vectorizer.transform(texts)

class TFIDFExtractor(FeatureExtractor):
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) Feature Extractor.
    Represents text by weighing terms based on their importance in the document and corpus.
    """
    def __init__(self):
        # Initialize TfidfVectorizer with English stop words removal
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def fit_transform(self, texts: pd.Series) -> Tuple[Any, Any]:
        """
        Fits the TfidfVectorizer and returns the TF-IDF matrix.
        """
        features = self.vectorizer.fit_transform(texts)
        return features, self.vectorizer

    def transform(self, texts: pd.Series) -> Any:
        """
        Transforms texts using the fitted TfidfVectorizer.
        """
        return self.vectorizer.transform(texts)

class NGramExtractor(FeatureExtractor):
    """
    N-Gram Feature Extractor.
    Captures local context by considering sequences of N words.
    Defaults to Bi-grams (n=2).
    """
    def __init__(self, n: int = 2):
        # Initialize CountVectorizer for n-grams with English stop words removal
        self.vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')

    def fit_transform(self, texts: pd.Series) -> Tuple[Any, Any]:
        """
        Fits the n-gram vectorizer and returns the feature matrix.
        """
        features = self.vectorizer.fit_transform(texts)
        return features, self.vectorizer

    def transform(self, texts: pd.Series) -> Any:
        """
        Transforms texts using the fitted n-gram vectorizer.
        """
        return self.vectorizer.transform(texts)
