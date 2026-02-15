import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import TextDataLoader
from src.features import BoWExtractor, TFIDFExtractor, NGramExtractor
from src.models import NBClassifier, SVMClassifier, LRClassifier
from src.evaluator import ModelEvaluator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. Load Data
    loader = TextDataLoader(target_categories=['sport', 'politics'])
    df = loader.load_data()
    
    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
    )
    
    # Define experiments
    feature_extractors = {
        'BoW': BoWExtractor(),
        'TF-IDF': TFIDFExtractor(),
        'N-Grams': NGramExtractor(n=2)
    }
    
    classifiers = {
        'Naive Bayes': NBClassifier(),
        'SVM': SVMClassifier(),
        'Logistic Regression': LRClassifier()
    }
    
    evaluator = ModelEvaluator()
    results = []
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # 2. Run Experiments
    for feat_name, extractor in feature_extractors.items():
        logging.info(f"Extracting features using {feat_name}...")
        X_train_feat, _ = extractor.fit_transform(X_train_text)
        X_test_feat = extractor.transform(X_test_text)
        
        for clf_name, clf in classifiers.items():
            logging.info(f"Training {clf_name} with {feat_name} features...")
            
            # Train
            clf.train(X_train_feat, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_feat)
            
            # Real Evaluation on Test
            metrics = evaluator.evaluate(y_test, y_pred, clf_name, feat_name)
            results.append(metrics)
            
            # Plot Confusion Matrix
            save_path = f"results/cm_{clf_name}_{feat_name}.png".replace(' ', '_')
            evaluator.plot_confusion_matrix(y_test, y_pred, labels=['sport', 'politics'], 
                                          title=f'Confusion Matrix: {clf_name} ({feat_name})', 
                                          save_path=save_path)

    # 3. Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/comparison_results.csv', index=False)
    
    print("\nFinal Results Summary:")
    print(results_df.sort_values(by='Accuracy', ascending=False))

if __name__ == "__main__":
    main()
