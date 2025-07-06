# Diabetes Detection Using ML (PIMA Indians Dataset)

A machine learning pipeline to predict diabetes in women using the PIMA Indians Diabetes Database. The project includes full EDA, feature exploration, model comparison, and hyperparameter tuning to select the most effective classifier based on Recall.

---

## Overview

This project tackles the binary classification problem of predicting whether a patient has diabetes based on medical diagnostic measurements such as glucose concentration, BMI, and blood pressure.

The complete pipeline includes:

- Exploratory Data Analysis (EDA)
- Feature exploration using KDE, box plots, and pair plots
- Model training with multiple classifiers
- Performance evaluation using multiple metrics
- Model tuning using RandomizedSearchCV
- Final model selection based on Recall (to minimize false negatives)

---

## Dataset

- Name: PIMA Indians Diabetes Dataset  
- Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Samples: 768
- Features: 8 (including glucose, insulin, BMI, age, etc.)
- Target: Binary (1 = Diabetic, 0 = Non-Diabetic)

---

## Tech Stack

- Python 3.10
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Jupyter Notebook

---

## Exploratory Data Analysis

- Handled missing or zero-value anomalies
- Visualizations used:
  - Box plots (outliers)
  - Pair plots (feature relations)
  - KDE plots (feature distributions)
  - Heatmap (feature correlation)

---

## Models Used

The following classifiers were trained and evaluated:

- RandomForestClassifier
- GradientBoostingClassifier
- SVC
- NuSVC
- LogisticRegression
- RidgeClassifier

Evaluation Metrics:
- Accuracy
- Precision
- Recall (priority metric)
- F1 Score
- ROC-AUC Curve
- Confusion Matrix

### Final Model
- RidgeClassifier
- Tuned using RandomizedSearchCV
- Chosen based on best Recall (to prioritize true positives for diabetes detection)

---

## How to Run

### 1. Clone the repository
bash

git clone https://github.com/Vernnit/Diabetes_Detection.git

cd Diabetes_Detection


### 2. Set up environment

bash

conda env create -f requirements.yaml

conda activate diabetes-detection


### 3. Run the notebook
bash
jupyter notebook notebooks/diabetes_detection.ipynb


---

## Future Improvements

- Test on external datasets (cross-hospital validation)
- Incorporate ensemble stacking for better generalization

---

## Author

- Vernnit Verma (https://github.com/Vernnit)

---
