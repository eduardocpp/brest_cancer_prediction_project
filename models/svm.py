from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale", random_state=42):
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_svm(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, preds
