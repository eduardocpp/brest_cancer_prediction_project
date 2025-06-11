from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

def train_loess_approx(X_train, y_train):
    model = RidgeClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_loess(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, preds
