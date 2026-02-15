# Text Classification: Sport vs. Politics

This repository contains the solution for the Text Classification assignment. It implements a binary classifier to distinguish between "Sport" and "Politics" news articles using the BBC News dataset.

## Project Overview

The project compares three Machine Learning algorithms:
1.  Naive Bayes
2.  Support Vector Machine (SVM)
3.  Logistic Regression

And three feature representation techniques:
1.  Bag of Words (BoW)
2.  TF-IDF
3.  N-Grams (Bi-grams)

## Directory Structure
```
.
├── src/
│   ├── data_loader.py    # Class to load and clean data
│   ├── features.py       # Classes for Feature Extraction
│   ├── models.py         # Classes for ML Classifiers
│   └── evaluator.py      # Class for Evaluation metrics
├── data/                 # Directory for dataset (auto-downloaded)
├── results/              # Directory for output metrics and plots
├── main.py               # Main execution script
├── requirements.txt      # Project dependencies
└── report.md             # Detailed assignment report
```

## Setup and Installation

1.  **Clone the repository** (if applicable).
2.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/Mac
    # venv\Scripts\activate   # On Windows
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to download data, train models, and generate results:

```bash
python main.py
```

This will:
1.  Download the BBC dataset (if not present).
2.  Filter for 'sport' and 'politics' categories.
3.  Train all model combinations.
4.  Print a summary table.
5.  Save detailed metrics to `results/comparison_results.csv`.
6.  Save confusion matrix plots to `results/`.

## Results
See `report.md` for a detailed analysis of the results.
In summary, Naive Bayes and SVM (with TF-IDF) achieved near-perfect accuracy on the test set.

## Requirements
-   Python 3.x
-   scikit-learn
-   pandas
-   numpy
-   matplotlib
-   seaborn
-   requests
