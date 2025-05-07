# XAI Techniques for Different Machine Learning Models

This repository contains the implementation of various Explainable AI (XAI) techniques applied to different types of machine learning models including binary classification, anomaly detection, and clustering.

## Project Overview

This project investigates how different Explainable AI (XAI) techniques can enhance the interpretability and transparency of various machine learning tasks. The study applies XAI methods to three core machine learning tasks using real-world datasets:

1. **Binary Classification**: Predicting Chronic Kidney Disease using medical attributes
2. **Anomaly Detection**: Detecting credit card fraud in financial transactions
3. **Clustering**: Segmenting retail customers based on purchasing behavior

By integrating methods such as Permutation Feature Importance, Anchor Explanations, Counterfactuals, SHAP, and LIME, the project demonstrates how to improve model transparency and trustworthiness across different domains.

## Datasets Used

- **Classification**: Chronic Kidney Disease dataset from UCI Machine Learning Repository
- **Anomaly Detection**: Credit Card Fraud Detection dataset from Kaggle
- **Clustering**: Online Retail dataset from UCI Machine Learning Repository

## Key XAI Techniques Implemented

### Classification (CKD Dataset)
- Permutation Feature Importance
- Anchor Explanations
- Counterfactual Explanations (using DiCE)

### Anomaly Detection (Credit Card Fraud)
- SHAP (SHapley Additive exPlanations)
  - Summary plots
  - Waterfall plots for individual predictions
- LIME (Local Interpretable Model-agnostic Explanations)

### Clustering (Online Retail)
- Feature Contribution Analysis
- Cluster-wise Feature Variance Analysis
- RFM (Recency, Frequency, Monetary) segmentation with explainability

## Repository Structure

```
├── data/
│   ├── chronic_kidney_disease.csv
│   ├── creditcard.csv
│   └── OnlineRetail.csv
├── notebooks/
│   ├── 1_Classification_XAI.ipynb
│   ├── 2_Anomaly_Detection_XAI.ipynb
│   └── 3_Clustering_XAI.ipynb
├── src/
│   ├── utils/
│   │   ├── preprocessing.py
│   │   ├── visualization.py
│   │   └── xai_methods.py
│   ├── classification/
│   │   └── kidney_disease_classifier.py
│   ├── anomaly_detection/
│   │   └── fraud_detector.py
│   └── clustering/
│       └── customer_segmentation.py
├── results/
│   ├── classification_results/
│   ├── anomaly_detection_results/
│   └── clustering_results/
├── requirements.txt
├── README.md
└── report.pdf
```

## Installation & Setup

1. Clone this repository:
```bash
git clone https://github.com/bhagyapriyyaaa/xai-techniques.git
cd xai-techniques
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Key Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- shap
- lime
- dice-ml
- alibi (for Anchor explanations)

## Usage

Each notebook in the `notebooks/` directory contains a complete pipeline for a specific ML task with integrated XAI techniques:

1. **Classification with XAI**:
```bash
jupyter notebook notebooks/1_Classification_XAI.ipynb
```

2. **Anomaly Detection with XAI**:
```bash
jupyter notebook notebooks/2_Anomaly_Detection_XAI.ipynb
```

3. **Clustering with XAI**:
```bash
jupyter notebook notebooks/3_Clustering_XAI.ipynb
```

## Key Findings

### Classification
- Specific gravity (sg), serum creatinine (sc), and packed cell volume (pcv) are the top contributors to CKD prediction accuracy
- Anchor explanations provide rule-based interpretations with over 99% precision
- Counterfactuals reveal that improving albumin levels, RBC count, and controlling hypertension can significantly influence CKD prediction outcomes

### Anomaly Detection
- SHAP analysis identifies features V24, V12, and V22 as having the highest impact on fraud detection
- LIME explanations provide localized understanding of individual transaction classifications

### Clustering
- Frequency of purchases is the strongest contributor to customer segmentation
- Three distinct customer segments were identified:
  - Cluster 0: Low-value, potentially churned customers
  - Cluster 1: Mid-value, regular customers
  - Cluster 2: High-value, loyal customers
