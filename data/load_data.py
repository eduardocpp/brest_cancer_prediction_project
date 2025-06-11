from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def load_and_prepare_data(test_size=0.2, random_state=1):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target).map({0: "M", 1: "B"})  # 0 = malignant, 1 = benign


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True), X.columns.tolist()
