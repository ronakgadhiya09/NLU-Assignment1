import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, List
import logging

class ModelEvaluator:
    """
    Evaluates model performance using standard metrics.
    """

    def evaluate(self, y_true: Any, y_pred: Any, model_name: str, feature_name: str) -> Dict[str, Any]:
        """
        Calculates accuracy, precision, recall, and F1-score.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            model_name: Name of the model (e.g., 'Naive Bayes').
            feature_name: Name of the feature set (e.g., 'BoW').
            
        Returns:
            Dict containing the metrics.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'Model': model_name,
            'Feature': feature_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        logging.info(f"Results for {model_name} with {feature_name}: {metrics}")
        return metrics

    def plot_confusion_matrix(self, y_true: Any, y_pred: Any, labels: List[str], title: str, save_path: str):
        """
        Generates and saves a confusion matrix plot.
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Confusion matrix saved to {save_path}")
