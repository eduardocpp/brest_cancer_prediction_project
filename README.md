BRCA Machine Learning Analysis 🧬

This project performs a complete analysis of breast cancer biopsy data using Principal Component Analysis (PCA) and a variety of supervised machine learning models. It covers feature scaling, dimensionality reduction, model training and evaluation, ensemble methods, and rich visualizations.

📊 Dataset

The dataset used is the brca dataset from the dslabs package. It contains numeric predictors derived from biopsy images of breast tissue and a corresponding label for each sample indicating whether the tumor is benign (B) or malignant (M).

🧳 Main Steps

Data Preprocessing: Normalization and stratified train-test split.

PCA: Principal Component Analysis reduces the dimensionality to 10 components while preserving variance.

Model Training: Multiple supervised learning models are trained using PCA-transformed data.

Evaluation: Accuracy, confusion matrix, and ROC curves are used for performance evaluation.

Ensemble Method: A majority vote mechanism combines predictions from all models.

Visualization: PCA projections, feature importance charts, model performance, and architecture visualizations are provided.

🎯 Supervised Learning

Supervised learning is a machine learning technique where a model is trained on a labeled dataset. Each example in the dataset includes input features and an output label. The goal is for the model to learn the relationship between inputs and outputs, so it can predict the label of unseen data. This is particularly useful in medical diagnoses, where the goal is to predict outcomes like the presence of a disease.

⚙️ Principal Component Analysis (PCA)

PCA is a statistical method used for dimensionality reduction. It transforms the original variables into a new set of orthogonal components (principal components), ordered by the amount of variance they explain in the data. By keeping only the top components, we reduce computational complexity and noise while retaining most of the information. PCA is especially important in high-dimensional datasets like this one, helping prevent overfitting and improving model generalization.

📊 Models Used

Below are brief explanations of the models implemented in this project:

Logistic Regression (GLM)

A linear classifier that models the log-odds of the binary outcome using a logistic function. It is simple, interpretable, and effective for linearly separable data.

k-Nearest Neighbors (kNN)

This algorithm classifies each sample based on the majority class among its k nearest training examples. It is intuitive and works well with well-separated data but is sensitive to the choice of k and scaling.

Random Forest

An ensemble method based on decision trees. Each tree is trained on a random subset of data and features, and the final decision is a majority vote. It is robust to overfitting and handles non-linearities well.

Support Vector Machine (SVM)

SVM finds the optimal hyperplane that maximally separates the two classes in the feature space. It is effective in high-dimensional spaces and with clear class boundaries.

Decision Tree

A model that recursively splits the data into subsets based on feature values. Although easy to interpret, single trees can overfit without pruning or regularization.

Multi-Layer Perceptron (MLP)

A type of artificial neural network composed of layers of interconnected neurons. It is capable of learning complex, non-linear relationships but requires careful tuning and more data.

Gradient Boosting

A sequential ensemble method where each new tree corrects the residuals of the previous one. It excels in structured datasets and is often one of the top-performing models in machine learning competitions.

Loess Approximation (Ridge Regression Version)

A local regression technique that smooths data by fitting multiple low-degree polynomials. Due to compatibility reasons, this is approximated using ridge regression to capture local trends.

🛃 Installation

To run this project locally, clone the repository and install the dependencies using:

pip install -r requirements.txt

📅 Author

Developed by Eduardo Rios, based on the HarvardX PH525.3x course.

🌐 LinkedIn: https://www.linkedin.com/in/eduardo-rios-de-andrade-sousa-6a108b260

📸 Instagram: @eduardo.rios19
