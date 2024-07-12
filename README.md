# Mechanism of Action (MoA) Prediction

## Project Overview
This project aims to predict a molecule's biological activity, specifically its Mechanism of Action (MoA), using various machine learning algorithms. The MoA of a drug refers to the biological process by which it produces its therapeutic effects. This work is based on a dataset provided by The Connectivity Map, in partnership with MIT, Harvard, LISH, and the NIH.

## Dataset
The dataset, sourced from the Laboratory for Innovation Science at Harvard via a Kaggle competition, combines gene expression and cell viability data. It provides insights into the activity of genes and the responses of cells to various drugs across 100 different cell types.

### Key Features:
- 23,814 rows and 875 features
- Discrete attributes: 'cp_type', 'cp_time', 'cp_dose'

## Methodology
We approached this as a multi-label classification problem, implementing and comparing several machine learning models:

1. Naive Bayes (Baseline)
2. Logistic Regression
3. Support Vector Machines (SVM)
4. Decision Trees
5. Random Forest
6. Artificial Neural Networks (ANN)
7. K-Nearest Neighbors (KNN)
8. Convolutional Neural Networks (CNN)

### Preprocessing Steps:
- One-hot encoding for discrete attributes
- Feature scaling and standardization
- Principal Component Analysis (PCA) for dimensionality reduction
- Sampling of the dataset to improve computational efficiency

## Results
Model performances were evaluated using log loss. Key findings:
- CNN demonstrated the best performance on unseen data with a testing loss of 0.01657
- Logistic Regression, SVM, and ANN showed strong performance
- Random Forest outperformed individual Decision Trees
- Naive Bayes, Decision Trees, and KNN served as baseline models with comparatively poor results

## Conclusion
This study contributes novel algorithms for MoA prediction and provides a clear rationale for parameter selections, enhancing the interpretability and applicability of the proposed models in the domain of Mechanism of Action prediction.

## Future Work
- Implement additional deep learning models
- Further optimize hyperparameters
- Explore advanced feature engineering techniques
