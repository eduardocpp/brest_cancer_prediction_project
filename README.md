# BRCA Machine Learning Analysis 🧬

This project performs a complete analysis of breast cancer biopsy data using Principal Component Analysis (PCA) and a variety of supervised machine learning models. It covers feature scaling, dimensionality reduction, model training and evaluation, ensemble methods, and rich visualizations.

## 📊 Dataset

The dataset used is the `brca` dataset from the `dslabs` package. It contains numeric predictors derived from biopsy images of breast tissue and a corresponding label for each sample indicating whether the tumor is **benign (B)** or **malignant (M)**.

## 🧳 Main Steps

* **Data Preprocessing**: Normalization and stratified train-test split.
* **PCA**: Principal Component Analysis reduces the dimensionality to 10 components while preserving variance.
* **Model Training**: Multiple supervised learning models are trained using PCA-transformed data.
* **Evaluation**: Accuracy, confusion matrix, and ROC curves are used for performance evaluation.
* **Ensemble Method**: A majority vote mechanism combines predictions from all models.
* **Visualization**: PCA projections, feature importance charts, model performance, and architecture visualizations are provided.

## 🎯 Supervised Learning

Supervised learning is a machine learning technique in which a model is trained on a labeled dataset. Each training example includes input features and a corresponding output label. The model learns to map inputs to outputs so that it can accurately predict labels for new, unseen data. This approach is particularly effective in domains like medical diagnostics, where reliable prediction is critical.

## ⚙️ Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a statistical technique for reducing the dimensionality of large datasets. It transforms the original variables into a new set of uncorrelated variables called principal components. These components are ordered by the amount of variance they explain in the data. By selecting only the top components, PCA helps reduce noise, avoid overfitting, and improve model generalization — especially in high-dimensional spaces like gene expression or medical imaging.

## 📊 Models Used

Below are detailed explanations of the models implemented in this project:

### Logistic Regression (GLM)

A linear model that estimates the probability of a binary outcome using the logistic function. It is widely used for its simplicity and interpretability and performs well when the relationship between features and the target is linear.

### k-Nearest Neighbors (kNN)

kNN classifies new instances based on the majority label of their *k* nearest neighbors in the training set. It is a non-parametric method that is simple and effective, though it can be sensitive to the choice of *k* and computationally expensive on large datasets.

### Random Forest

An ensemble learning method that constructs multiple decision trees and merges their outputs. By averaging many trees trained on different subsets of data and features, it improves accuracy and reduces overfitting.

### Support Vector Machine (SVM)

SVM aims to find the hyperplane that best separates classes in a high-dimensional space. It is effective for classification tasks with clear margins and is robust in complex and high-dimensional spaces.

### Decision Tree

A flowchart-like model that makes decisions based on feature values. Each node represents a condition, and each branch represents an outcome. Though intuitive, decision trees are prone to overfitting and are often used as base learners in ensembles.

### Multi-Layer Perceptron (MLP)

MLP is a type of feedforward neural network that uses layers of nodes (neurons) to learn complex relationships. With one or more hidden layers and non-linear activation functions, it is capable of modeling highly non-linear data patterns.

### Gradient Boosting

An advanced ensemble method that builds trees sequentially, where each tree tries to correct errors made by the previous one. This method is highly effective on structured/tabular data and is widely used in winning solutions to machine learning competitions.

### Loess Approximation (Ridge Regression Version)

LOESS is a non-parametric regression method that fits simple models to localized subsets of data. In this project, it is approximated using ridge regression, offering smooth predictions while preventing overfitting through L2 regularization.

## 🛃 Installation

To run this project locally, clone the repository and install the dependencies using:

```bash
pip install -r requirements.txt
```

## 📅 Author

Developed by **Eduardo Rios**.

* 🌐 [LinkedIn](https://www.linkedin.com/in/eduardoribeirodata)
* 📸 [Instagram](https://www.instagram.com/eduardo.ribeiro.ai)

---

Feel free to contribute or open issues in this repository!
