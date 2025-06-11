from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_random_forest(X_train, y_train, mtry=5, random_state=9):
    model = RandomForestClassifier(max_features=mtry, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_rf(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, preds

def get_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    importance_dict = dict(zip(feature_names, importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_importance[:top_n]
