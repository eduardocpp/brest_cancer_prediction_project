from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def train_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def evaluate_gradient_boosting(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, preds