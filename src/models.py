from abc import ABC, abstractmethod
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from typing import Any

class TextClassifier(ABC):
    """
    Abstract base class for text classifiers.
    Defines the standard interface for training and predicting.
    """

    @abstractmethod
    def train(self, X_train: Any, y_train: Any):
        """
        Trains the underlying model logic using the provided features and labels.
        
        Args:
            X_train (Any): The feature matrix for training data.
            y_train (Any): The labels for training data.
        """
        pass

    @abstractmethod
    def predict(self, X_test: Any) -> Any:
        """
        Predicts labels for the provided test features.
        
        Args:
            X_test (Any): The feature matrix for test data.
            
        Returns:
            Any: Predicted labels.
        """
        pass

class NBClassifier(TextClassifier):
    """
    Naive Bayes Classifier (Multinomial).
    Suitable for classification with discrete features (e.g., word counts).
    """
    def __init__(self):
        # MultinomialNB is standard for text classification
        self.model = MultinomialNB()

    def train(self, X_train: Any, y_train: Any):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: Any) -> Any:
        return self.model.predict(X_test)

class SVMClassifier(TextClassifier):
    """
    Support Vector Machine Classifier.
    Effective in high-dimensional spaces, commonly used for text classification.
    """
    def __init__(self):
        # Using linear kernel which is generally best for high-dimensional text data
        # and computationally more efficient than non-linear kernels.
        self.model = SVC(kernel='linear') 

    def train(self, X_train: Any, y_train: Any):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: Any) -> Any:
        return self.model.predict(X_test)

class LRClassifier(TextClassifier):
    """
    Logistic Regression Classifier.
    A baseline probabilistic classifier that works well for binary/multiclass classification.
    """
    def __init__(self):
        # max_iter increased to ensure convergence for larger datasets
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train: Any, y_train: Any):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: Any) -> Any:
        return self.model.predict(X_test)
